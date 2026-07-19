# transforms_hecktor.py
import torch
from monai import transforms as T

from deterministic_transforms import KEYS


def random_transforms(patch=96):
    """Transformaciones aleatorias (se aplican on-the-fly en cada época). Opera
    sobre "ct"/"pt"/"label" por separado (no sobre "image") porque la augmentation
    de intensidad debe aplicar SOLO a CT (fiel al paper); la fusión a "image" de
    2 canales ocurre al final, después de augmentar."""
    return T.Compose([
        # La caché guarda "ct"/"pt" en float16 (ver deterministic_transforms.load_transforms).
        # RandAffined usa grid_sample, que no soporta float16 en CPU (los workers del
        # DataLoader corren en CPU), así que subimos a float32 antes de augmentar. El costo
        # es solo por-parche, no por todo el volumen cacheado.
        T.EnsureTyped(keys=["ct", "pt"], dtype=torch.float32),

        # Parche centrado en clases foreground: 0.45 tumor, 0.45 ganglios, 0.1 fondo
        T.RandCropByLabelClassesd(
            keys=["ct", "pt", "label"], label_key="label",
            spatial_size=(patch, patch, patch),
            ratios=[0.1, 0.45, 0.45], num_classes=3, num_samples=1,
        ),
        # Augmentations geométricas (paper: Affine + Flip) — mismas en los 3 canales
        T.RandAffined(keys=["ct", "pt", "label"], prob=0.2,
                      rotate_range=(0.26, 0.26, 0.26), scale_range=(0.1, 0.1, 0.1),
                      mode=("bilinear", "bilinear", "nearest")),
        T.RandFlipd(keys=["ct", "pt", "label"], prob=0.5, spatial_axis=0),
        T.RandFlipd(keys=["ct", "pt", "label"], prob=0.5, spatial_axis=1),
        T.RandFlipd(keys=["ct", "pt", "label"], prob=0.5, spatial_axis=2),
        # Augmentations de intensidad — paper: SOLO CT
        T.RandScaleIntensityd(keys=["ct"], factors=0.1, prob=0.3),
        T.RandShiftIntensityd(keys=["ct"], offsets=0.1, prob=0.3),
        T.RandGaussianNoised(keys=["ct"], prob=0.2, std=0.05),
        T.RandGaussianSmoothd(keys=["ct"], prob=0.2),
        # Fusión al final -> "image" de 2 canales
        T.ConcatItemsd(keys=["ct", "pt"], name="image", dim=0),
        T.EnsureTyped(keys=["image", "label"]),
    ])


# Mantener compatibilidad con versiones anteriores
def preprocessing(train=True, patch=192):
    """Legacy: retorna todas las transformaciones en un solo Compose.
    Usar solo con CacheDataset o Dataset normal, NO con PersistentDataset."""
    xf = [
        T.LoadImaged(keys=KEYS),
        T.EnsureChannelFirstd(keys=KEYS),
        T.Orientationd(keys=KEYS, axcodes="RAS"),
        T.Spacingd(keys=["ct", "pt"], pixdim=(1, 1, 1), mode="bilinear"),
        T.Spacingd(keys=["label"],   pixdim=(1, 1, 1), mode="nearest"),
        T.CropForegroundd(keys=KEYS, source_key="pt", allow_smaller=True),
        T.ScaleIntensityRanged(keys=["ct"], a_min=-200, a_max=400,
                               b_min=0.0, b_max=1.0, clip=False),
        T.NormalizeIntensityd(keys=["pt"], nonzero=True, channel_wise=True),
        T.Lambdad(keys=["ct", "pt"], func=lambda x: 1.0 / (1.0 + (-x).exp())),
        T.ConcatItemsd(keys=["ct", "pt"], name="image", dim=0),
    ]
    if train:
        xf += [
            T.RandCropByLabelClassesd(
                keys=["image", "label"], label_key="label",
                spatial_size=(patch, patch, patch),
                ratios=[0.1, 0.45, 0.45], num_classes=3, num_samples=1,
            ),
            T.RandAffined(keys=["image", "label"], prob=0.2,
                          rotate_range=(0.26, 0.26, 0.26), scale_range=(0.1, 0.1, 0.1),
                          mode=("bilinear", "nearest")),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            T.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3, channel_wise=False),
            T.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            T.RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            T.RandGaussianSmoothd(keys=["image"], prob=0.2),
        ]
    xf += [T.EnsureTyped(keys=["image", "label"])]
    return T.Compose(xf)
