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

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {dev}")

net = SegResNetDS(spatial_dims=3, init_filters=INIT_FILTERS, in_channels=2,
                  out_channels=3, blocks_down=BLOCKS_DOWN, norm="batch",
                  act="relu", dsdepth=DSDEPTH).to(dev)
loss_fn = DeepSupervisionLoss(DiceCELoss(softmax=True, to_onehot_y=True),
                              weight_mode="exp")
opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

x = torch.randn(BATCH, 2, PATCH, PATCH, PATCH, device=dev)
y = torch.randint(0, 3, (BATCH, 1, PATCH, PATCH, PATCH), device=dev).float()

if dev == "cuda":
    torch.cuda.reset_peak_memory_stats()
net.train(); opt.zero_grad()
try:
    with torch.amp.autocast("cuda", enabled=USE_AMP):
        out = net(x)                 # lista de salidas (supervisión profunda)
        loss = loss_fn(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
    if dev == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"OK  patch={PATCH} filtros={INIT_FILTERS} amp={USE_AMP} batch={BATCH}"
              f"  ->  pico {peak:.2f} GB  (loss {loss.item():.4f})")
    else:
        print(f"OK (CPU) patch={PATCH} filtros={INIT_FILTERS} amp={USE_AMP} batch={BATCH}"
              f"  ->  loss {loss.item():.4f}")
except torch.cuda.OutOfMemoryError:
    print(f"OOM patch={PATCH} filtros={INIT_FILTERS} amp={USE_AMP} batch={BATCH}"
          f"  ->  no cabe, baja un escalón (ver §7)")
except Exception as e:
    print(f"Error: {e}")
