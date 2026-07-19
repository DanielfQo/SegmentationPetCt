# make_datalist.py
#
# Paso 2 del pipeline: escanea el dataset YA preprocesado (ver
# deterministic_transforms.py, paso 1) y arma datalist_1fold.json con el
# split train/val/test. No toca el dataset crudo ni hace ningún cómputo:
# solo empareja archivos y asigna folds.
import json
import os
import random

from deterministic_transforms import OUTPUT_ROOT


def discover_cases(output_root):
    """Escanea `output_root`/imagesTr/labelsTr (formato fijo escrito por
    deterministic_transforms.save_case: {id}_0000.nii.gz, {id}_0001.nii.gz,
    labelsTr/{id}.nii.gz) y arma {case_id: {"image": [ct, pt], "label": lbl}}."""
    images_dir = os.path.join(output_root, "imagesTr")
    labels_dir = os.path.join(output_root, "labelsTr")

    cases_dict = {}
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return cases_dict

    suffix_ct = "_0000.nii.gz"
    for name in sorted(os.listdir(images_dir)):
        if not name.endswith(suffix_ct):
            continue
        case_id = name[: -len(suffix_ct)]
        ct_path = os.path.join(images_dir, name)
        pt_path = os.path.join(images_dir, f"{case_id}_0001.nii.gz")
        lbl_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        if os.path.exists(pt_path) and os.path.exists(lbl_path):
            cases_dict[case_id] = {"image": [ct_path, pt_path], "label": lbl_path}

    return cases_dict


def build_datalist(cases_dict, seed=0):
    """Baraja y separa los casos: 70% train / 15% val / 15% test."""
    cases = sorted(list(cases_dict.keys()))
    random.seed(seed)
    random.shuffle(cases)

    n_test = int(len(cases) * 0.15)
    n_val = int(len(cases) * 0.15)
    test_cases = cases[:n_test]
    val_cases = cases[n_test:n_test + n_val]
    train_cases = cases[n_test + n_val:]

    datalist = {
        "training": [{**cases_dict[c], "fold": 0} for c in val_cases]      # fold 0 = validación
                   + [{**cases_dict[c], "fold": 1} for c in train_cases],  # fold 1 = entrenamiento
        "testing": [{**cases_dict[c]} for c in test_cases],
    }
    return datalist, train_cases, val_cases, test_cases


def main():
    output_root = str(OUTPUT_ROOT)
    print(f"Buscando dataset preprocesado en: {output_root}")

    cases_dict = discover_cases(output_root)
    if not cases_dict:
        raise FileNotFoundError(
            f"No se encontraron casos preprocesados en {output_root}. "
            f"Ejecuta 'python deterministic_transforms.py' primero."
        )

    datalist, train_cases, val_cases, test_cases = build_datalist(cases_dict)

    json.dump(datalist, open("datalist_1fold.json", "w"), indent=2)
    print(f"Total detectados: {len(cases_dict)} casos -> "
          f"train {len(train_cases)} / val {len(val_cases)} / test {len(test_cases)}")


if __name__ == "__main__":
    main()
