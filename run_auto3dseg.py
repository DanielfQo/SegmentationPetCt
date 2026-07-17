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
