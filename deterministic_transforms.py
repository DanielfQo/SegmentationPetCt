# deterministic_transforms.py
#
# Paso 1 del pipeline: preprocesamiento determinista de HECKTOR (carga,
# reorientación, resampleo, recorte por foreground y normalización). Corre
# UNA vez sobre todo el dataset crudo y escribe los resultados como NIfTI
# reales en disco -inspeccionables con cualquier visor-, en vez de un cache
# binario opaco. Separado de random_transforms() (data augmentation, en
# transforms_hecktor.py) para poder correrlo, revisarlo y reanudarlo de
# forma independiente del entrenamiento.
#
# Pipeline (correr en este orden desde modelos/segNet):
#   1) python deterministic_transforms.py   <- este script: descubre los
#      casos del dataset crudo y escribe, por caso, imagesTr/{id}_0000.nii.gz
#      (CT), imagesTr/{id}_0001.nii.gz (PET) y labelsTr/{id}.nii.gz (label)
#      ya resampleados/recortados/normalizados bajo OUTPUT_ROOT. Reanudable:
#      si la salida de un caso ya existe, se lo salta (--force para redoer).
#   2) python make_datalist.py              <- escanea OUTPUT_ROOT y arma
#      datalist_1fold.json con el split train/val/test.
#   3) python train_1fold.py                <- entrena leyendo
#      datalist_1fold.json; solo carga los NIfTI ya preprocesados
#      (load_transforms) y aplica random_transforms() on-the-fly.
import argparse
import os
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import transforms as T
from monai.transforms import MapTransform, SpatialCrop

KEYS = ["ct", "pt", "label"]

# ----------------------------- Rutas -----------------------------
# Dataset crudo: varias ubicaciones candidatas, relativas al CWD o al script.
RAW_RELATIVE_DIRS = [
    "HECKTOR2025_raw",
    "../HECKTOR2025_raw",
    "../../HECKTOR2025_raw",
    "../../../HECKTOR2025_raw",
    "HECKTOR 2025 Training Data",
    "../HECKTOR 2025 Training Data",
    "../../HECKTOR 2025 Training Data",
    "data",
    "../data",
    "../../data",
]

# Salida del preprocesamiento determinista (paso 1). make_datalist.py (paso 2) lee de aquí.
# Hermana de HECKTOR2025_raw en la raíz del repo (deterministic_transforms.py vive en
# modelos/segNet/, así que hacen falta 3 .parent para llegar a la raíz).
OUTPUT_ROOT =  "./HECKTOR2025_preprocessed"

# Cache de train_1fold.py para load_transforms() (solo IO+concat: ya es barato porque
# lo pesado -resample/crop/normalización- quedó resuelto en OUTPUT_ROOT).
CACHE_DIR = "./persistent_cache"


def find_raw_dataset_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for r_dir in RAW_RELATIVE_DIRS:
        if os.path.exists(r_dir) and os.path.isdir(r_dir):
            return os.path.abspath(r_dir)
        script_rel = os.path.join(script_dir, r_dir)
        if os.path.exists(script_rel) and os.path.isdir(script_rel):
            return os.path.abspath(script_rel)
    return os.path.abspath("../../HECKTOR2025_raw")


def discover_raw_cases(dataset_root):
    """Recorre subcarpetas por paciente del dataset CRUDO y arma
    {case_id: {"ct": ct_path, "pt": pt_path, "label": label_path}}.
    Soporta el formato 2022 ({id}__CT/__PT/{id}.nii.gz) y 2025
    ({id}_0000/_0001/{id}.nii.gz)."""
    cases_dict = {}
    for item in sorted(os.listdir(dataset_root)):
        item_path = os.path.join(dataset_root, item)
        if not os.path.isdir(item_path):
            continue

        ct_2022 = os.path.join(item_path, f"{item}__CT.nii.gz")
        pt_2022 = os.path.join(item_path, f"{item}__PT.nii.gz")
        ct_2025 = os.path.join(item_path, f"{item}_0000.nii.gz")
        pt_2025 = os.path.join(item_path, f"{item}_0001.nii.gz")
        lbl = os.path.join(item_path, f"{item}.nii.gz")

        if os.path.exists(ct_2022) and os.path.exists(pt_2022) and os.path.exists(lbl):
            cases_dict[item] = {"ct": ct_2022, "pt": pt_2022, "label": lbl}
        elif os.path.exists(ct_2025) and os.path.exists(pt_2025) and os.path.exists(lbl):
            cases_dict[item] = {"ct": ct_2025, "pt": pt_2025, "label": lbl}

    return cases_dict


class CropHeadAndNeckd(MapTransform):
    """Recorta una caja fija de tamaño físico (mm) anclada al tope de la cabeza,
    centrada en la línea media del cuerpo. Asume orientación RAS y spacing 1 mm
    (1 voxel = 1 mm; eje 2 = z crece hacia superior, tope de cabeza = z máximo).

    IMPORTANTE: debe aplicarse ANTES de la normalización de intensidad, cuando el CT
    aún está en HU y el PET en sus valores crudos (SUV/actividad).

    Usa SpatialCrop (no slicing directo) porque es lo que MONAI usa para mantener
    el affine del MetaTensor consistente con el nuevo origen tras el recorte; un
    slice manual (`tensor[:, x0:x1, ...]`) NO actualiza el affine y dejaría el
    NIfTI guardado con el origen físico del volumen sin recortar (verificado).

    Antes de recortar, guarda en d["_hn_pre_crop_label_counts"] los voxeles de
    tumor (clase 1) y ganglios (clase 2) que había ANTES del recorte, para que
    preprocess_case() pueda detectar no solo el caso "eliminado por completo"
    (ya cubierto por el conteo final == 0) sino el caso más sutil de una lesión
    que queda CORTADA: sobreviven algunos voxeles pero no todos."""
    def __init__(self, keys, head_key="pt", body_key="ct", label_key="label",
                 size_mm=(200, 200, 310), body_thr=-500, head_band_mm=60):
        super().__init__(keys)
        self.head_key, self.body_key, self.label_key = head_key, body_key, label_key
        self.sx, self.sy, self.sz = size_mm
        self.body_thr = body_thr
        self.head_band_mm = head_band_mm

    def __call__(self, data):
        d = dict(data)
        body = (d[self.body_key][0] > self.body_thr)          # (X,Y,Z) máscara de cuerpo (HU)
        pet  = d[self.head_key][0]                            # (X,Y,Z) PET crudo

        # Tope de la cabeza: slice z más alta con señal PET relevante (cerebro).
        pet_thr = pet.mean() + pet.std()
        z_present = (pet > pet_thr).any(0).any(0)             # (Z,) bool
        nz = torch.where(z_present)[0]
        z_top = int(nz.max()) if len(nz) else int(body.shape[2]) - 1
        z0 = max(0, z_top - self.sz)
        z1 = z_top

        # Línea media (x,y): centroide del cuerpo en una banda ANGOSTA (head_band_mm,
        # ~ancho de una cabeza) pegada al tope, NO en toda la altura [z0:z1]. Si el FOV
        # nativo es más corto que size_mm (CHUM-001: solo ~295mm de Z, pedimos 310mm),
        # z0 se clampea a 0 y [z0:z1] termina siendo casi todo el cuerpo -torso y hombros
        # incluidos-; promediar la extensión lateral sobre esa franja tan alta arrastra
        # el centro hacia los hombros (más anchos que el cuello) y desplaza la caja lo
        # suficiente como para cortar el tumor. Verificado en CHUM-001: con la franja
        # completa cy=209 (corta el tumor en Y); con una banda de 60mm cy=299-300 (el
        # tumor entero queda adentro). Independiente de self.sz para no heredar el
        # mismo problema si la caja completa también termina clampeada.
        slab_z0 = max(z0, z1 - self.head_band_mm)
        slab = body[:, :, slab_z0:z1]
        xs = torch.where(slab.any(2).any(1))[0]
        ys = torch.where(slab.any(2).any(0))[0]
        cx = int((xs.min() + xs.max()) // 2) if len(xs) else body.shape[0] // 2
        cy = int((ys.min() + ys.max()) // 2) if len(ys) else body.shape[1] // 2
        x0 = max(0, cx - self.sx // 2); x1 = x0 + self.sx
        y0 = max(0, cy - self.sy // 2); y1 = y0 + self.sy

        label = d.get(self.label_key)
        if label is not None:
            d["_hn_pre_crop_label_counts"] = {c: int((label == c).sum()) for c in (1, 2)}
            fg = (label[0] > 0)
            idx = torch.where(fg)
            if len(idx[0]):
                print(f"  label bbox: x[{int(idx[0].min())},{int(idx[0].max())}] "
                      f"y[{int(idx[1].min())},{int(idx[1].max())}] "
                      f"z[{int(idx[2].min())},{int(idx[2].max())}]")
                print(f"  crop  box : x[{x0},{x1}] y[{y0},{y1}] z[{z0},{z1}]  (z_top={z_top})")

        cropper = SpatialCrop(roi_start=[x0, y0, z0], roi_end=[x1, y1, z1])
        for k in self.key_iterator(d):
            d[k] = cropper(d[k])
        return d


# ----------------------------- Cómputo (en memoria) -----------------------------
def compute_transforms():
    """Transformaciones pesadas y deterministas: carga, reorientación,
    resampleo, recorte a la región H&N y normalización. Deja "ct"/"pt"/"label"
    como volúmenes separados (sin concatenar) listos para guardarse a NIfTI."""
    return T.Compose([
        T.LoadImaged(keys=KEYS),
        T.EnsureChannelFirstd(keys=KEYS),
        T.Orientationd(keys=KEYS, axcodes="RAS"),

        # (a) CT define la grilla de referencia (1 mm isotrópico). PET y label se
        # remuestrean a la grilla EXACTA del CT ya resampleado (tamaño+spacing+origen+
        # dirección) con ResampleToMatchd, NO con un Spacingd independiente: el PET
        # crudo de HECKTOR trae su propio origen/FOV, distinto del CT (verificado en
        # CHUM-001: ~26 mm de diferencia en X/Y). Un Spacingd(keys=["ct","pt"]) por
        # separado iguala el spacing pero deja cada modalidad centrada en su propio
        # origen -CT y PET quedan desalineados en el NIfTI final, aunque el shape
        # coincida-. Mismo patrón que preprocesamiento2.py (resample_to_reference).
        T.Spacingd(keys=["ct"], pixdim=(1, 1, 1), mode="bilinear"),
        T.ResampleToMatchd(keys=["pt"],    key_dst="ct", mode="bilinear"),
        T.ResampleToMatchd(keys=["label"], key_dst="ct", mode="nearest"),

        # (b) Recorte a la silueta del cuerpo por CT (limpia aire/camilla). Va después
        # de Spacingd porque ahí CT/PT/label ya comparten grilla; antes de resamplear no
        # tienen el mismo shape y el bounding box no se podría aplicar por igual a las 3 keys.
        T.CropForegroundd(keys=KEYS, source_key="ct",
                           select_fn=lambda x: x > -500, allow_smaller=True),

        # (c) Recorte REAL a la región H&N: caja fija (200x200x310 mm, como el paper)
        # anclada al tope de la cabeza detectado por señal PET. Antes de esto,
        # CropForegroundd(source_key="pt") con el select_fn por defecto (x>0) no servía:
        # el PET es positivo en casi todo el cuerpo (cerebro/corazón/hígado/riñones/
        # vejiga), así que devolvía la caja del cuerpo entero. Va ANTES de normalizar
        # (usa HU crudo del CT y SUV/actividad crudo del PET).
        CropHeadAndNeckd(keys=KEYS, head_key="pt", body_key="ct",
                         size_mm=(200, 200, 310), body_thr=-500),

        # (d) Normalización por canal (fiel al paper):
        T.ScaleIntensityRanged(keys=["ct"], a_min=-200, a_max=400,
                               b_min=0.0, b_max=1.0, clip=False),
        T.NormalizeIntensityd(keys=["pt"], nonzero=True, channel_wise=True),
        T.Lambdad(keys=["ct", "pt"], func=lambda x: 1.0 / (1.0 + (-x).exp())),

        # (e) Tamaño uniforme: si el recorte quedó más chico que la caja (borde del
        # cuerpo/volumen), rellena hasta 200x200x310 (post-sigmoid, 0 = fuera/fondo).
        T.SpatialPadd(keys=KEYS, spatial_size=(200, 200, 310)),
    ])


# ----------------------------- Guardado a NIfTI -----------------------------
# CT y PET salen de compute_transforms() en (0, 1) (por el sigmoid final). NIfTI no
# soporta float16 directamente (nibabel/ITK rechazan ese dtype), así que para pesar
# la mitad que float32 usamos uint16: al fijar set_data_dtype, nibabel calcula un
# scl_slope/scl_inter que ajusta el rango real de cada volumen a uint16 y lo guarda
# en el header; cualquier lector (nibabel, SimpleITK, MONAI LoadImaged) lo revierte
# solo. Error de reconstrucción verificado con datos reales: <= 1e-5 absoluto.
def _meta_tensor_to_array_affine(meta_tensor):
    """MetaTensor (1,H,W,D) -> (array (H,W,D) float32, affine 4x4 numpy)."""
    arr = meta_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    affine = meta_tensor.affine.detach().cpu().numpy()
    return arr, affine


def _save_intensity_nifti(meta_tensor, path):
    """Guarda un volumen normalizado en (0,1) como NIfTI uint16 escalado
    (scl_slope/scl_inter en el header). Cualquier lector estándar (nibabel,
    SimpleITK, MONAI LoadImaged) devuelve el float reconstruido de forma
    transparente, sin que el resto del pipeline necesite saberlo."""
    arr, affine = _meta_tensor_to_array_affine(meta_tensor)
    img = nib.Nifti1Image(arr, affine)
    img.header.set_data_dtype(np.uint16)
    nib.save(img, f"{path}.nii.gz")


def _save_label_nifti(meta_tensor, path):
    """Guarda la máscara de clases como NIfTI uint8 (valores exactos, sin escalar)."""
    arr, affine = _meta_tensor_to_array_affine(meta_tensor)
    img = nib.Nifti1Image(arr.astype(np.uint8), affine)
    nib.save(img, f"{path}.nii.gz")


def save_case(case_id, transformed, output_root):
    """Guarda ct/pt/label ya procesados (MetaTensors, con affine correcto tras
    reorientar/resamplear/recortar) como imagesTr/{case_id}_0000.nii.gz (CT),
    imagesTr/{case_id}_0001.nii.gz (PET) y labelsTr/{case_id}.nii.gz (label)
    bajo `output_root`.

    Escritura atómica por caso: los 3 archivos se escriben primero a un directorio
    temporal (dentro de output_root, mismo filesystem) y solo se mueven a su lugar
    final cuando los 3 están completos. Así, si el proceso se cae a mitad de un
    caso (CT escrito, PET no), imagesTr/labelsTr nunca queda con una salida a medias
    que además pase el chequeo de "ya procesado" por error."""
    images_dir = Path(output_root) / "imagesTr"
    labels_dir = Path(output_root) / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(output_root)) as tmp:
        tmp = Path(tmp)
        _save_intensity_nifti(transformed["ct"], tmp / f"{case_id}_0000")
        _save_intensity_nifti(transformed["pt"], tmp / f"{case_id}_0001")
        _save_label_nifti(transformed["label"], tmp / case_id)

        shutil.move(str(tmp / f"{case_id}_0000.nii.gz"), str(images_dir / f"{case_id}_0000.nii.gz"))
        shutil.move(str(tmp / f"{case_id}_0001.nii.gz"), str(images_dir / f"{case_id}_0001.nii.gz"))
        shutil.move(str(tmp / f"{case_id}.nii.gz"), str(labels_dir / f"{case_id}.nii.gz"))


_LABEL_CLASS_NAMES = {1: "tumor", 2: "ganglios"}


def preprocess_case(case_id, files, transform, output_root):
    """Corre el pipeline para un caso, cuenta voxeles de tumor (clase 1) y
    ganglios (clase 2) en la máscara YA recortada -para detectar si el recorte
    H&N se llevó puesto el GTV- y guarda. También compara contra los conteos
    PRE-recorte que deja CropHeadAndNeckd (d["_hn_pre_crop_label_counts"]) para
    detectar el caso de una lesión CORTADA (sobreviven algunos voxeles, pero no
    todos) y no solo el de una lesión eliminada por completo. Alerta por
    stdout apenas lo detecta.

    Devuelve (n_tumor, n_nodes, truncated): truncated es un dict
    {"tumor"|"ganglios": (antes, despues, pct_perdido)} con las clases cortadas
    (vacío si ninguna lo fue)."""
    transformed = transform(files)
    lbl = transformed["label"]
    n_tumor = int((lbl == 1).sum())
    n_nodes = int((lbl == 2).sum())

    pre_crop = transformed.get("_hn_pre_crop_label_counts", {})
    after_by_class = {1: n_tumor, 2: n_nodes}
    truncated = {}
    for cls, name in _LABEL_CLASS_NAMES.items():
        before = pre_crop.get(cls)
        after = after_by_class[cls]
        if before is not None and before > 0 and after < before:
            pct = 100.0 * (before - after) / before
            truncated[name] = (before, after, pct)
            estado = "eliminado por completo" if after == 0 else "INCOMPLETO (quedan algunos voxeles)"
            print(f"[ALERTA] {case_id}: el recorte H&N cortó {name} "
                  f"({before} -> {after} voxeles, -{pct:.1f}%)  <-- {estado}")

    save_case(case_id, transformed, output_root)
    return n_tumor, n_nodes, truncated


# ----------------------------- Carga del resultado (para entrenamiento) -----------------------------
def load_transforms():
    """Carga los NIfTI YA preprocesados (ver save_case), SIN concatenar: deja
    "ct"/"pt"/"label" separados porque random_transforms() (transforms_hecktor.py)
    necesita augmentar la intensidad SOLO en "ct" (fiel al paper) antes de fusionar
    en "image". Solo IO -sin resample/crop/normalización, eso ya quedó resuelto al
    guardar-, por eso alcanza con cachear esto en train_1fold.py (ver CACHE_DIR) en
    vez de recalcularlo cada época."""
    return T.Compose([
        T.LoadImaged(keys=KEYS),
        T.EnsureChannelFirstd(keys=KEYS),
        T.EnsureTyped(keys=["ct", "pt"], dtype=torch.float16),
        T.EnsureTyped(keys=["label"]),
    ])


def val_transforms():
    """Validación: carga + fusión directa en "image" (sin augmentation, así que
    a diferencia de load_transforms() sí concatena acá mismo)."""
    return T.Compose([
        T.LoadImaged(keys=KEYS),
        T.EnsureChannelFirstd(keys=KEYS),
        T.ConcatItemsd(keys=["ct", "pt"], name="image", dim=0),
        T.EnsureTyped(keys=["image"], dtype=torch.float32),   # float32 por robustez del sliding window
        T.EnsureTyped(keys=["label"]),
    ])


# ----------------------------- CLI batch -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Paso 1: preprocesamiento determinista HECKTOR -> NIfTI")
    parser.add_argument("--input-root", type=str, default=None,
                         help="Carpeta del dataset crudo (por defecto se autodetecta)")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo los primeros N casos")
    parser.add_argument("--case", type=str, default=None, help="Procesar un unico case_id")
    parser.add_argument("--force", action="store_true", help="Reprocesar aunque ya exista la salida")
    args = parser.parse_args()

    dataset_root = args.input_root or find_raw_dataset_root()
    print(f"Dataset crudo: {dataset_root}")
    print(f"Salida:        {args.output_root}")

    cases_dict = discover_raw_cases(dataset_root)
    if args.case is not None:
        cases_dict = {k: v for k, v in cases_dict.items() if k == args.case}
        if not cases_dict:
            raise ValueError(f"case_id '{args.case}' no encontrado bajo {dataset_root}")

    case_ids = sorted(cases_dict.keys())
    if args.limit is not None:
        case_ids = case_ids[:args.limit]

    if not case_ids:
        print(f"No se encontraron casos en {dataset_root}")
        return

    images_dir = Path(args.output_root) / "imagesTr"
    labels_dir = Path(args.output_root) / "labelsTr"

    def _is_done(case_id):
        return (
            (images_dir / f"{case_id}_0000.nii.gz").exists()
            and (images_dir / f"{case_id}_0001.nii.gz").exists()
            and (labels_dir / f"{case_id}.nii.gz").exists()
        )

    already_done = {c for c in case_ids if _is_done(c)}
    to_process = case_ids if args.force else [c for c in case_ids if c not in already_done]

    print(f"Total casos: {len(case_ids)}  Ya procesados: {len(case_ids) - len(to_process)}  "
          f"A procesar: {len(to_process)}")

    transform = compute_transforms()
    n_ok, n_err = 0, 0
    empty_cases = []
    truncated_cases = []
    for i, case_id in enumerate(to_process, start=1):
        try:
            n_t, n_n, truncated = preprocess_case(case_id, cases_dict[case_id], transform, args.output_root)
            n_ok += 1
            flags = []
            if n_t == 0 and n_n == 0:
                empty_cases.append(case_id)
                flags.append("SIN ETIQUETAS")
            if truncated:
                truncated_cases.append(case_id)
                flags.append("LESION CORTADA")
            flag = ("  <-- " + " / ".join(flags)) if flags else ""
            print(f"[{i}/{len(to_process)}] {case_id} OK  (tumor={n_t} nodes={n_n}){flag}")
        except Exception as exc:  # noqa: BLE001 - se registra y se sigue con el resto del dataset
            n_err += 1
            print(f"[{i}/{len(to_process)}] {case_id} ERROR: {exc}")

    print(f"\nListo: {n_ok} casos guardados, {n_err} errores. Salida en: {args.output_root}")

    if empty_cases:
        print(f"\nATENCION: {len(empty_cases)} casos quedaron sin etiquetas tras el recorte:")
        print("  " + ", ".join(empty_cases))
        print("  Revisa si el recorte H&N los tiro (arreglar) o si son casos sin GTV (el "
              "paper menciona que a veces falta el tumor).")

    if truncated_cases:
        print(f"\nATENCION: {len(truncated_cases)} casos con tumor/ganglios CORTADOS por el "
              f"recorte H&N (quedan incompletos, no vacíos del todo -detalle arriba en cada [ALERTA]):")
        print("  " + ", ".join(truncated_cases))
        print("  Sube size_mm o revisa DOWN_MM/pet_thr para esos casos.")


if __name__ == "__main__":
    main()
