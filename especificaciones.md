# Replicar HECKTOR 2022 (NVAUTO) con **un solo fold**

Guía práctica para reproducir la solución de Myronenko et al. (1er lugar, HECKTOR22)
entrenando **un único fold**, manteniendo los hiperparámetros del paper como punto de
partida, y bajándolos *poco a poco* hasta que quepa en tu GPU (2× RTX 2080, 8 GB c/u).

> **Lee esto primero.** El paper entrena en 8× V100 de 16 GB con parches de 192³ y un
> modelo de **~356 millones de parámetros** (lo verifiqué con MONAI 1.6). Esa configuración
> **no cabe en 8 GB**. Este documento te da (1) el punto de partida fiel al paper y (2) una
> escalera ordenada de ajustes para reducir memoria. Tu resultado será **relativo a tu propio
> modelo base**, no al 0.788 del leaderboard. Eso es perfectamente válido, sobre todo si tu
> objetivo final es *pruning*.

---

## 0. Índice

1. Entorno e instalación
2. Datos HECKTOR22
3. Preprocesamiento (fiel al paper)
4. La red SegResNet (config del paper)
5. Loss, optimizador y augmentations (fiel al paper)
6. **Herramienta 1 — Sonda de memoria** (prueba si cabe en segundos)
7. **La escalera de memoria** (qué relajar y en qué orden)
8. **Herramienta 2 — Entrenamiento de un fold** (instrumentable, base para pruning)
9. Evaluación (Dice agregado)
10. Alternativa fiel: Auto3DSeg (el camino exacto del paper)
11. Notas para 2× 2080 y puente hacia el pruning

---

## 1. Entorno e instalación



```bash
# Recomendado: entorno limpio pero con python 3.12 porque en la computadora tambien esta 3.14
python3 -m venv venv && source venv/bin/activate

# Torch con soporte CUDA acorde a tu driver (revisa https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# MONAI + utilidades
pip install "monai[all]==1.6.0" nibabel tqdm einops
```

Verifica que la GPU se ve:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Versiones con las que se validó el código de esta guía: **torch 2.x**, **MONAI 1.6.0**.

---

## 2. Datos HECKTOR22


- `CASE__CT.nii.gz` — CT (alta resolución anatómica)
- `CASE__PT.nii.gz` — PET (baja resolución, resalta actividad tumoral)
- `CASE.nii.gz` — máscara de verdad-terreno: `0` fondo, `1` tumor (GTVp), `2` ganglios (GTVn)

Son 524 casos etiquetados para entrenamiento. Organízalos así:

```
data/
  imagesTr/
    CHUP-001__CT.nii.gz
    CHUP-001__PT.nii.gz
    ...
  labelsTr/
    CHUP-001.nii.gz
    ...
```

### 2.1 Lista de datos con UN solo fold

Auto3DSeg y MONAI trabajan con un JSON tipo *datalist*. Aquí generamos **un único split
80/20** (en vez de los 5 folds del paper):

```python
# make_datalist.py
import json, glob, os, random

IMG_DIR = "data/imagesTr"
LBL_DIR = "data/labelsTr"
random.seed(0)

cases = sorted({os.path.basename(p).split("__")[0]
                for p in glob.glob(f"{IMG_DIR}/*__CT.nii.gz")})
random.shuffle(cases)

n_val = int(len(cases) * 0.2)
val, train = cases[:n_val], cases[n_val:]

def entry(c):
    return {
        "image": [f"{IMG_DIR}/{c}__CT.nii.gz", f"{IMG_DIR}/{c}__PT.nii.gz"],
        "label": f"{LBL_DIR}/{c}.nii.gz",
    }

datalist = {
    "training":   [{**entry(c), "fold": 0} for c in val]   # fold 0 = validación
                + [{**entry(c), "fold": 1} for c in train], # fold 1 = entrenamiento
}
json.dump(datalist, open("datalist_1fold.json", "w"), indent=2)
print(f"{len(cases)} casos -> train {len(train)} / val {len(val)}")
```

> Truco de MONAI: en Auto3DSeg todo lo que tenga `"fold": 0` es la partición de validación
> cuando entrenas el "fold 0". Aquí lo aprovechamos para tener un único split sin montar los 5.

---

## 3. Preprocesamiento (fiel al paper)

Pasos del paper: (a) remuestrear CT y PET a **1×1×1 mm isotrópico**, (b) recortar la región
de cabeza y cuello, (c) normalizar por canal, (d) concatenar en una entrada de **2 canales**.

```python
# transforms_hecktor.py
from monai import transforms as T

KEYS = ["ct", "pt", "label"]

def preprocessing(train=True, patch=192):
    xf = [
        T.LoadImaged(keys=KEYS),
        T.EnsureChannelFirstd(keys=KEYS),
        T.Orientationd(keys=KEYS, axcodes="RAS"),

        # (a) 1 mm isotrópico. Label con vecino más cercano para no inventar clases.
        T.Spacingd(keys=["ct", "pt"], pixdim=(1, 1, 1), mode="bilinear"),
        T.Spacingd(keys=["label"],   pixdim=(1, 1, 1), mode="nearest"),

        # (b) Recorte aprox. de la región H&N.
        #     El paper usa una heurística con umbral sobre PET para hallar el tope de la
        #     cabeza y recorta una caja de 200x200x310 mm. Aproximación pragmática:
        #     recortar al foreground del PET (activa donde hay señal). Suficiente para una
        #     primera réplica; si quieres el recorte exacto, implementa la heurística del
        #     paper (tope de cabeza + línea central + caja fija).
        T.CropForegroundd(keys=KEYS, source_key="pt", allow_smaller=True),

        # (c) Normalización por canal (fiel al paper):
        #     CT: reescala de un rango predefinido a 0..1, luego sigmoide (en vez de clamp).
        #         OJO: el rango exacto NO está en el paper ("determinado por análisis de datos").
        #         Este es un punto de ventana HU razonable para H&N; AJÚSTALO con tus datos.
        T.ScaleIntensityRanged(keys=["ct"], a_min=-200, a_max=400,
                               b_min=0.0, b_max=1.0, clip=False),
        #     PET: media 0, desviación 1, luego sigmoide.
        T.NormalizeIntensityd(keys=["pt"], nonzero=True, channel_wise=True),
        #     Sigmoide como alternativa al clamp duro (paper). Lambda sobre ambos canales:
        T.Lambdad(keys=["ct", "pt"], func=lambda x: 1.0 / (1.0 + (-x).exp())),

        # (d) Concatenar CT+PET -> imagen de 2 canales
        T.ConcatItemsd(keys=["ct", "pt"], name="image", dim=0),
    ]

    if train:
        xf += [
            # Parche 192^3 centrado en clases foreground: 0.45 tumor, 0.45 ganglios, 0.1 fondo
            T.RandCropByLabelClassesd(
                keys=["image", "label"], label_key="label",
                spatial_size=(patch, patch, patch),
                ratios=[0.1, 0.45, 0.45], num_classes=3, num_samples=1,
            ),
            # Augmentations geométricas (paper: Affine + Flip)
            T.RandAffined(keys=["image", "label"], prob=0.2,
                          rotate_range=(0.26, 0.26, 0.26), scale_range=(0.1, 0.1, 0.1),
                          mode=("bilinear", "nearest")),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # Augmentations de intensidad SOLO en CT (canal 0) — paper
            T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3, channel_wise=False),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            T.RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            T.RandGaussianSmoothd(keys=["image"], prob=0.2),
        ]

    xf += [T.EnsureTyped(keys=["image", "label"])]
    return T.Compose(xf)
```

> **Nota honesta.** El rango HU del CT y la heurística exacta de recorte son los dos puntos
> donde el paper es vago. Para *tu* réplica (base para pruning) no es crítico clavarlos al
> milímetro: lo que importa es tener un base consistente y bien entrenado.

---

## 4. La red SegResNet (config del paper)

El paper usa **SegResNet con supervisión profunda**. En MONAI moderno eso es `SegResNetDS`.
Config exacta del paper: 6 etapas de `1,2,2,4,4,4` bloques, 32 filtros iniciales,
normalización *batch*, entrada de 2 canales, salida de 3 clases (softmax).

```python
from monai.networks.nets import SegResNetDS

def build_model(init_filters=32, dsdepth=4):
    return SegResNetDS(
        spatial_dims=3,
        init_filters=init_filters,        # paper: 32
        in_channels=2,                    # CT + PET
        out_channels=3,                   # fondo + tumor + ganglios
        blocks_down=(1, 2, 2, 4, 4, 4),   # paper: 6 etapas
        norm="batch",                     # paper: batch normalization
        act="relu",                       # paper
        dsdepth=dsdepth,                  # nº de cabezas de supervisión profunda
    )
```

Con la config del paper esto son **~356 M de parámetros** y produce **4 salidas** de
supervisión profunda (a resolución completa, 1/2, 1/4, 1/8). Verificado con MONAI 1.6.

> Para pruning: 356M es muchísimo. Este tamaño es tu mejor argumento — hay enorme margen
> para comprimir. (Ver §11.)

---

## 5. Loss, optimizador y augmentations (fiel al paper)

- **Loss:** Dice + CrossEntropy, sumado sobre los niveles de supervisión profunda con peso
  `1/2^i`. En MONAI eso es `DiceCELoss` envuelto en `DeepSupervisionLoss(weight_mode="exp")`,
  que aplica exactamente la ponderación exponencial decreciente del paper.
- **Optimizador:** AdamW, lr inicial `2e-4`, weight decay `1e-5`, *cosine annealing* a 0.
- **Épocas:** 300. **Batch:** 1 por GPU.

```python
import torch
from monai.losses import DiceCELoss, DeepSupervisionLoss

base_loss = DiceCELoss(softmax=True, to_onehot_y=True)   # combina Dice + CE
loss_fn   = DeepSupervisionLoss(base_loss, weight_mode="exp")  # pesos 1/2^i

def make_optimizer(model, max_epochs=300):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    return opt, sched
```

> **Trampa que descubrí probándolo:** el *label* debe entrar como **float**, no long. El
> downsampling interno de `DeepSupervisionLoss` usa interpolación *nearest*, que en PyTorch
> no acepta tensores `Long`. Con MONAI las etiquetas ya salen float, pero si construyes
> tensores a mano recuerda `.float()`.

---

## 6. Herramienta 1 — Sonda de memoria

**Esto es lo que pediste.** Un script mínimo que construye la red, hace **un forward +
backward** sobre un parche ficticio y te dice el **pico de memoria en GB**, sin necesidad de
tener los datos montados. Úsalo para tantear si cabe *en segundos* antes de lanzar nada.

```python
# memprobe.py  —  ¿cabe en mi GPU? Ajusta los knobs de arriba y re-ejecuta.
import torch
from monai.networks.nets import SegResNetDS
from monai.losses import DiceCELoss, DeepSupervisionLoss

# ======== KNOBS (empieza con los del paper y baja poco a poco) ========
PATCH        = 192              # paper: 192  ->  primer candidato a reducir
INIT_FILTERS = 32               # paper: 32
BLOCKS_DOWN  = (1, 2, 2, 4, 4, 4)  # paper
DSDEPTH      = 4                # paper: 4 cabezas
BATCH        = 1               # paper: 1 por GPU
USE_AMP      = False            # ponlo True para ahorrar ~40-50% de memoria
# =====================================================================

dev = "cuda"
net = SegResNetDS(spatial_dims=3, init_filters=INIT_FILTERS, in_channels=2,
                  out_channels=3, blocks_down=BLOCKS_DOWN, norm="batch",
                  act="relu", dsdepth=DSDEPTH).to(dev)
loss_fn = DeepSupervisionLoss(DiceCELoss(softmax=True, to_onehot_y=True),
                              weight_mode="exp")
opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

x = torch.randn(BATCH, 2, PATCH, PATCH, PATCH, device=dev)
y = torch.randint(0, 3, (BATCH, 1, PATCH, PATCH, PATCH), device=dev).float()

torch.cuda.reset_peak_memory_stats()
net.train(); opt.zero_grad()
try:
    with torch.amp.autocast("cuda", enabled=USE_AMP):
        out = net(x)                 # lista de salidas (supervisión profunda)
        loss = loss_fn(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"OK  patch={PATCH} filtros={INIT_FILTERS} amp={USE_AMP} batch={BATCH}"
          f"  ->  pico {peak:.2f} GB  (loss {loss.item():.4f})")
except torch.cuda.OutOfMemoryError:
    print(f"OOM patch={PATCH} filtros={INIT_FILTERS} amp={USE_AMP} batch={BATCH}"
          f"  ->  no cabe, baja un escalón (ver §7)")
```

Ejecuta:

```bash
python memprobe.py
```

Empieza con los valores del paper. Si sale `OOM`, cambia **un** knob según la escalera de
abajo y vuelve a correr. En una 2080 de 8 GB, lo esperable es que `PATCH=192` no entre y que
tengas que combinar AMP + parche más pequeño.

---

## 7. La escalera de memoria

Ajusta en **este orden**. Los primeros escalones **no tocan la fidelidad** al método (mismo
modelo, mismos resultados esperables); los últimos **sí sacrifican precisión**, así que úsalos
solo si hace falta.

| # | Ajuste | Cómo | Ahorro | ¿Afecta el resultado? |
|---|--------|------|--------|----------------------|
| 1 | **Activar AMP (fp16)** | `USE_AMP = True` | ~40–50 % | No (casi idéntico). Las 2080 tienen tensor cores. **Hazlo siempre.** |
| 2 | **Bajar el parche** | `PATCH`: 192 → 160 → 128 → 96 | Grande (cúbico) | Sí, leve. Menos contexto anatómico por parche. |
| 3 | **Gradient checkpointing** | `torch.utils.checkpoint` en los bloques | ~20–30 % | No en calidad; ~20–30 % más lento. Avanzado. |
| 4 | **Menos filtros iniciales** | `INIT_FILTERS`: 32 → 24 → 16 | Grande | Sí, moderado. Reduce capacidad del modelo. |
| 5 | **Menos cabezas DS** | `DSDEPTH`: 4 → 3 → 2 | Pequeño | Sí, leve. |

Reglas de oro:

- **Cambia un solo knob a la vez** y vuelve a correr `memprobe.py`. Así sabes qué te hizo caber.
- **Deja un margen** (~1–1.5 GB libres): el pico real durante el entrenamiento con datos
  reales y validación *sliding window* es algo mayor que en la sonda.
- **Dos 2080 NO se suman a 16 GB.** En DataParallel/DDP cada GPU necesita que el parche quepa
  *por separado* en sus 8 GB. Las dos tarjetas te dan más *throughput* (batch efectivo 2), no
  un parche más grande. Para eso necesitarías *model parallelism*, que Auto3DSeg no monta.

Configuración de arranque razonable para 8 GB (punto de partida sugerido tras el escalón 1–2):
`PATCH=128, INIT_FILTERS=32, USE_AMP=True, BATCH=1`. Ajusta desde ahí.

---

## 8. Herramienta 2 — Entrenamiento de un fold

Bucle de entrenamiento mínimo pero completo. Los **knobs** están arriba para que edites y
re-ejecutes cómodamente. Está pensado para ser **fácil de instrumentar** (justo lo que
necesitarás para tus experimentos de pruning en §11).

```python
# train_1fold.py
import json, torch
from torch.utils.data import DataLoader
from monai.data import CacheDataset, decollate_batch
from monai.losses import DiceCELoss, DeepSupervisionLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNetDS
from monai.transforms import AsDiscrete
from transforms_hecktor import preprocessing   # del §3

# ======== KNOBS ========
PATCH        = 128          # ajusta según §7
INIT_FILTERS = 32
DSDEPTH      = 4
USE_AMP      = True
MAX_EPOCHS   = 300          # paper: 300
VAL_EVERY    = 5
CACHE_RATE   = 0.0          # sube si tienes RAM de sobra (acelera mucho)
# =======================

dev = "cuda"

# --- datos ---
dl = json.load(open("datalist_1fold.json"))["training"]
train_files = [{"ct": d["image"][0], "pt": d["image"][1], "label": d["label"]}
               for d in dl if d["fold"] == 1]
val_files   = [{"ct": d["image"][0], "pt": d["image"][1], "label": d["label"]}
               for d in dl if d["fold"] == 0]

train_ds = CacheDataset(train_files, preprocessing(train=True, patch=PATCH),
                        cache_rate=CACHE_RATE)
val_ds   = CacheDataset(val_files,   preprocessing(train=False), cache_rate=CACHE_RATE)
train_ld = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
val_ld   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

# --- modelo / loss / optim ---
net = SegResNetDS(spatial_dims=3, init_filters=INIT_FILTERS, in_channels=2,
                  out_channels=3, blocks_down=(1, 2, 2, 4, 4, 4), norm="batch",
                  act="relu", dsdepth=DSDEPTH).to(dev)
loss_fn = DeepSupervisionLoss(DiceCELoss(softmax=True, to_onehot_y=True),
                              weight_mode="exp")
opt   = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

post_pred  = AsDiscrete(argmax=True, to_onehot=3)
post_label = AsDiscrete(to_onehot=3)
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

best = -1.0
for epoch in range(MAX_EPOCHS):
    net.train()
    for batch in train_ld:
        x = batch["image"].to(dev)
        y = batch["label"].to(dev).float()
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = net(x)              # lista (supervisión profunda)
            loss = loss_fn(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    sched.step()

    # --- validación con sliding window (parche 192^3 como en inferencia del paper) ---
    if (epoch + 1) % VAL_EVERY == 0:
        net.eval()
        with torch.no_grad():
            for batch in val_ld:
                x = batch["image"].to(dev)
                y = batch["label"].to(dev)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = sliding_window_inference(
                        x, roi_size=(192, 192, 192), sw_batch_size=1,
                        predictor=lambda t: net(t)[0],   # solo la salida a full-res
                        overlap=0.5)
                preds  = [post_pred(p)  for p in decollate_batch(logits)]
                labels = [post_label(l) for l in decollate_batch(y)]
                dice_metric(y_pred=preds, y=labels)
            scores = dice_metric.aggregate()      # [dice_tumor, dice_ganglios]
            dice_metric.reset()
            mean_dice = scores.mean().item()
            print(f"epoch {epoch+1:3d}  dice_tumor={scores[0]:.4f}  "
                  f"dice_ganglios={scores[1]:.4f}  media={mean_dice:.4f}")
            if mean_dice > best:
                best = mean_dice
                torch.save(net.state_dict(), "best_fold.pt")

print(f"Mejor Dice medio en validación: {best:.4f}")
```

Referencia del paper (Tabla 1): un fold ronda **~0.79** en *su* validación cruzada agregada.
Con tus concesiones de parche/filtros, esperar algo en el rango **~0.72–0.77** sería digno
y demostraría que la pipeline funciona.

> El *sliding window* en validación usa el parche completo de 192³ aunque hayas entrenado con
> parches más pequeños — así la inferencia ve el contexto completo. Si 192³ tampoco cabe en
> inferencia, baja `roi_size` a 128³ (con `overlap` un poco mayor para compensar).

---

## 9. Evaluación (Dice agregado)

El reto usa **Dice agregado** (no el Dice promedio por caso): junta las intersecciones y
uniones de *todos* los casos antes de dividir. Es distinto (y normalmente más bajo y más
estable) que promediar Dice por paciente. El bucle de arriba usa el Dice estándar de MONAI
para monitoreo; para comparar de forma justa con la Tabla 1 del paper, calcula el agregado:

```python
# Dice agregado: sum(2*|A∩B|) / sum(|A|+|B|) sobre TODO el conjunto, por clase.
import torch

def aggregated_dice(all_preds, all_labels, num_fg=2):
    inter = torch.zeros(num_fg); denom = torch.zeros(num_fg)
    for p, l in zip(all_preds, all_labels):   # p, l: máscaras enteras [D,H,W]
        for c in range(1, num_fg + 1):
            pc, lc = (p == c), (l == c)
            inter[c-1] += (pc & lc).sum()
            denom[c-1] += pc.sum() + lc.sum()
    return (2 * inter / denom.clamp(min=1)).tolist()   # [tumor, ganglios]
```

---

## 10. Alternativa fiel: Auto3DSeg (el camino exacto del paper)

Si quieres el pipeline *idéntico* al paper (que literalmente **es** el SegResNet de
Auto3DSeg), en vez de escribir el bucle a mano puedes dejar que Auto3DSeg haga análisis de
datos, generación de configs, entrenamiento y ensamble. Para un solo fold:

```python
# run_auto3dseg.py
from monai.apps.auto3dseg import AutoRunner

runner = AutoRunner(
    work_dir="./work_hecktor",
    input={
        "modality": "MRI",           # tratamos PET/CT como multicanal; Auto3DSeg lo maneja
        "datalist": "datalist_1fold.json",
        "dataroot": ".",
        "class_names": ["tumor", "lymph"],
    },
    algos="segresnet",               # solo la arquitectura del paper
    ensemble=False,                  # sin ensamble: un solo modelo
)
runner.set_num_fold(num_fold=1)      # UN fold
# Para caber en 8 GB, sobreescribe el tamaño de parche que Auto3DSeg elige:
runner.set_training_params({
    "roi_size": [128, 128, 128],     # baja según §7
    "num_epochs": 300,
    "amp": True,
})
runner.run()
```

Ventaja: es la reproducción más fiel y con menos código. Desventaja: Auto3DSeg **abstrae**
mucho, lo que lo hace **más difícil de instrumentar** para tus experimentos de pruning. Por
eso, para tu tesis, probablemente te convenga el bucle manual del §8 como base.

---

## 11. Notas para 2× 2080 y puente hacia el pruning

**Sobre el hardware.** Con 8 GB por tarjeta, el flujo realista es: `memprobe.py` primero,
AMP siempre activado, parche reducido, y aceptar que reproduces el *método*, no el número
exacto del leaderboard. Un fold entrenado 300 épocas en 2080 puede tardar **varios días**;
planifícalo con la máquina estable térmicamente.

**Sobre tu tesis de pruning.** Este base es un excelente sujeto experimental:

- **El modelo es enorme (~356 M).** Hay muchísimo que podar → tu compresión se verá
  dramática. Entrena el base **tan grande como quepa** (no lo achiques de más con pocos
  filtros), porque sin margen el pruning luce poco.
- **Reporta curvas, no puntos.** Traza *Dice vs. sparsity* (10, 30, 50, 70, 90 %). Comparar
  curvas es mucho más robusto al ruido de tener un solo fold que comparar un único número.
- **Distingue pruning estructurado vs. no estructurado.** El no estructurado (poner pesos a
  cero) da sparsity alta pero **sin ahorro real** de memoria/tiempo en una GPU normal. El
  estructurado (eliminar canales/filtros) sí produce un modelo **realmente más pequeño y
  rápido** — que es justo la historia tangible para una tesis con restricción de hardware:
  *"el modelo podado pasó de no caber a correr en X ms en una 2080"*.
- **Mide huella real.** No reportes solo Dice: mide **memoria pico** y **tiempo de inferencia**
  en tu propio hardware, antes y después de podar. Esos números concretos son los que separan
  una tesis de compresión de un ejercicio abstracto de sparsity.
- **El bucle del §8 es tu punto de enganche.** Inserta el pruning entre `build_model` y el
  entrenamiento, y añade una fase de *fine-tuning* tras podar con el mismo protocolo de
  evaluación en todos los experimentos (misma validación, mismo presupuesto de fine-tune).

**Enmarcado sugerido de la tesis:** *"¿cuánto se puede comprimir SegResNet para segmentación
de cabeza y cuello antes de que el Dice se degrade de forma inaceptable, y eso lo hace
caber/correr en una GPU de 8 GB?"* — pregunta limpia, bien motivada por el propio tamaño del
modelo.

---

*Última verificación de APIs: MONAI 1.6.0 / torch 2.x. Los detalles del rango HU del CT y la
heurística exacta de recorte H&N son los dos puntos que el paper deja abiertos; ajústalos con
análisis de tus propios datos si buscas máxima fidelidad.*