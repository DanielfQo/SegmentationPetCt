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
        T.CropForegroundd(keys=KEYS, source_key="pt", allow_smaller=True),

        # (c) Normalización por canal (fiel al paper):
        T.ScaleIntensityRanged(keys=["ct"], a_min=-200, a_max=400,
                               b_min=0.0, b_max=1.0, clip=False),
        T.NormalizeIntensityd(keys=["pt"], nonzero=True, channel_wise=True),
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
