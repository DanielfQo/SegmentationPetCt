# train_1fold.py
import json
import os

# Reduce fragmentación del allocator de CUDA (debe fijarse antes de crear contexto CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import torch
from torch.utils.data import DataLoader
from monai.data import PersistentDataset, decollate_batch
from monai.data.utils import pickle_hashing
from monai.losses import DiceCELoss, DeepSupervisionLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNetDS
from monai.transforms import AsDiscrete
from deterministic_transforms import CACHE_DIR, load_transforms, val_transforms
from transforms_hecktor import random_transforms

# ======== KNOBS ========
# Tu GPU tiene ~7.6 GiB. PATCH=128 + INIT_FILTERS=32 + DSDEPTH=4 no cupo (OOM en backward
# de la 1ra iteración). Corre memprobe.py para hallar la combinación más grande que sí quepa
# y actualiza estos valores en consecuencia (deja 1 escalón de margen para el pico real,
# que sí tiene overhead de dataloader/augmentation que el sweep sintético no captura).
PATCH        = 96
INIT_FILTERS = 32
DSDEPTH      = 4
USE_AMP      = True
MAX_EPOCHS   = 300          # paper: 300
VAL_EVERY    = 5
# =======================

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {dev}")

# --- datos ---
datalist_path = "datalist_1fold.json"
try:
    dl = json.load(open(datalist_path))["training"]
except FileNotFoundError:
    print(f"Error: No se encontró '{datalist_path}'. Ejecuta 'deterministic_transforms.py' "
          f"y luego 'make_datalist.py' primero.")
    exit(1)

train_files = [{"ct": d["image"][0], "pt": d["image"][1], "label": d["label"]}
               for d in dl if d["fold"] == 1]
val_files   = [{"ct": d["image"][0], "pt": d["image"][1], "label": d["label"]}
               for d in dl if d["fold"] == 0]

print(f"Train cases: {len(train_files)}, Val cases: {len(val_files)}")

# Lo pesado (resample/crop/normalización) ya quedó resuelto como NIfTI en
# HECKTOR2025_preprocessed por deterministic_transforms.py (paso 1). Acá solo se
# cachea la carga (load_transforms/val_transforms), barata pero no gratis si se repite
# cada época.
os.makedirs(CACHE_DIR, exist_ok=True)

# hash_transform: la clave de caché de PersistentDataset por defecto depende SOLO de las
# rutas de entrada, no del transform usado. load_transforms() (train, ct/pt separados) y
# val_transforms() (val, concatenado en "image") son distintos y comparten CACHE_DIR, así
# que se incluye el hash del transform para que un mismo caso nunca pueda leer del cache
# la salida del OTRO pipeline si algún día train/val dejaran de ser disjuntos.
train_ds = PersistentDataset(train_files, load_transforms(), cache_dir=CACHE_DIR,
                              hash_transform=pickle_hashing)
val_ds   = PersistentDataset(val_files,   val_transforms(), cache_dir=CACHE_DIR,
                              hash_transform=pickle_hashing)

# Wrapper que aplica augmentation al vuelo sobre los datos cacheados
random_xf = random_transforms(patch=PATCH)

class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper que aplica transformaciones aleatorias on-the-fly sobre un dataset cacheado."""
    def __init__(self, base_ds, transform):
        self.base_ds = base_ds
        self.transform = transform
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        data = self.base_ds[idx]
        result = self.transform(data)
        # RandCropByLabelClassesd con num_samples=1 devuelve una lista de 1 dict
        if isinstance(result, list):
            return result[0]
        return result

train_aug_ds = AugmentedDataset(train_ds, random_xf)
train_ld = DataLoader(train_aug_ds, batch_size=1, shuffle=True, num_workers=4)
val_ld   = DataLoader(val_ds,       batch_size=1, shuffle=False, num_workers=2)

# --- modelo / loss / optim ---
net = SegResNetDS(spatial_dims=3, init_filters=INIT_FILTERS, in_channels=2,
                  out_channels=3, blocks_down=(1, 2, 2, 4, 4, 4), norm="batch",
                  act="relu", dsdepth=DSDEPTH).to(dev)
# DiceCELoss: include_background=False para que la componente Dice solo mida
# clases foreground (tumor, ganglios); smooth_nr/smooth_dr evitan 0/0 -> NaN
# cuando un parche cae en puro fondo (frecuente con tumores pequeños).
base_loss = DiceCELoss(softmax=True, to_onehot_y=True,
                       include_background=False,
                       smooth_nr=1e-5, smooth_dr=1e-5,
                       lambda_dice=1.0, lambda_ce=1.0)
loss_fn = DeepSupervisionLoss(base_loss, weight_mode="exp")
opt   = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

post_pred  = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

# --- checkpoint / resume: siempre se retoma desde el último mejor guardado, no desde
# donde se cortó el entrenamiento (así un crash nunca deja el modelo en un estado peor) ---
ckpt_path = "best_checkpoint.pt"
best = -1.0
start_epoch = 0
resuming = os.path.exists(ckpt_path)
if resuming:
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    net.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    best = ckpt["best_dice"]
    start_epoch = ckpt["epoch"]
    print(f"Checkpoint encontrado: retomando desde el mejor guardado "
          f"(epoch {start_epoch}, dice {best:.4f}).")

# --- log de métricas ---
csv_file = "metrics_log.csv"
# Si venimos de un resume, mantenemos el log existente; si no, lo (re)creamos con cabeceras.
if not (resuming and os.path.exists(csv_file)):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "dice_tumor", "dice_ganglios", "mean_dice"])

for epoch in range(start_epoch, MAX_EPOCHS):
    net.train()
    epoch_loss = 0
    for batch in train_ld:
        x = batch["image"].to(dev)
        y = batch["label"].to(dev).float()
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = net(x)              # lista (supervisión profunda)
            loss = loss_fn(out, y)
        # Protección NaN: si un parche all-background genera NaN/Inf, se salta
        # el update de pesos para no corromper el modelo entero.
        if not torch.isfinite(loss):
            opt.zero_grad()
            continue
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        epoch_loss += loss.item()
    sched.step()
    
    mean_loss = epoch_loss / len(train_ld)
    print(f"epoch {epoch+1:3d}/{MAX_EPOCHS}  loss={mean_loss:.4f}")

    # --- validación con sliding window (parche 192^3 como en inferencia del paper) ---
    val_loss_str = ""
    val_tumor = ""
    val_ganglios = ""
    val_mean = ""
    
    if (epoch + 1) % VAL_EVERY == 0:
        net.eval()
        if dev == "cuda":
            torch.cuda.empty_cache()  # libera bloques cacheados del entrenamiento antes de stitchear
        val_loss_sum = 0.0
        val_loss_count = 0
        # loss_fn espera lista (deep supervision), pero sliding_window_inference devuelve
        # solo la salida full-res. Se usa la DiceCELoss base directamente para val.
        val_loss_fn = DiceCELoss(softmax=True, to_onehot_y=True,
                                 include_background=False,
                                 smooth_nr=1e-5, smooth_dr=1e-5)
        with torch.no_grad():
            for batch in val_ld:
                x = batch["image"].to(dev)
                y = batch["label"].to(dev)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = sliding_window_inference(
                        x, roi_size=(PATCH, PATCH, PATCH), sw_batch_size=1,
                        predictor=lambda t: net(t),   # solo la salida a full-res
                        overlap=0.5, sw_device=dev, device="cpu")  # stitching en CPU: el volumen
                # Val loss (DiceCE sin deep supervision, sobre el volumen completo)
                val_loss_sum += val_loss_fn(logits.float(), y.cpu().float()).item()
                val_loss_count += 1
                        # completo (3 canales, tamaño real del paciente) no compite por VRAM con el training
                preds  = [post_pred(p)  for p in decollate_batch(logits)]
                labels = [post_label(l) for l in decollate_batch(y.cpu())]
                dice_metric(y_pred=preds, y=labels)
            scores = dice_metric.aggregate()      # [dice_tumor, dice_ganglios]
            dice_metric.reset()
            mean_dice = scores.mean().item()
            mean_val_loss = val_loss_sum / max(val_loss_count, 1)
            val_loss_str = f"{mean_val_loss:.4f}"
            val_tumor = f"{scores[0]:.4f}"
            val_ganglios = f"{scores[1]:.4f}"
            val_mean = f"{mean_dice:.4f}"
            print(f"epoch {epoch+1:3d}  val_loss={val_loss_str}  dice_tumor={val_tumor}  "
                  f"dice_ganglios={val_ganglios}  media={val_mean}")
            if mean_dice > best:
                best = mean_dice
                torch.save(net.state_dict(), "best_fold.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_dice": best,
                }, ckpt_path)
                print("Guardado nuevo mejor modelo.")

    # Guardar en CSV
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, f"{mean_loss:.4f}", val_loss_str, val_tumor, val_ganglios, val_mean])

print(f"Mejor Dice medio en validación: {best:.4f}")
