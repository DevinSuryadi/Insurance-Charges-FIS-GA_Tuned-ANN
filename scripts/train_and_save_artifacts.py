import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import torch

from general.model_utils import (
    N_AGE_MF,
    N_BMI_MF,
    N_SMOKER_MF,
    run_full_pipeline,
    save_json,
    serializable_fis_params,
)


REQUIRED_ARTIFACTS = [
    "preprocessing.joblib",
    "manual_fis_params.json",
    "ga_fis_params.json",
    "ann_model.pt",
    "metrics.json",
]


def main(force_retrain=False):
    project_dir = ROOT_DIR
    artifact_dir = project_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    existing = [name for name in REQUIRED_ARTIFACTS if (artifact_dir / name).exists()]
    if len(existing) == len(REQUIRED_ARTIFACTS) and not force_retrain:
        print("Artifacts already exist. Skip retraining.")
        print("Use --force if you want to retrain from scratch.")
        return

    print("Starting full training pipeline...")
    result = run_full_pipeline(csv_path=str(project_dir / "insurance.csv"))

    data = result["data"]
    manual_fis = result["manual_fis"]
    ga_fis = result["ga_fis"]
    ann_model = result["ann_model"]

    joblib.dump(
        {
            "scaler_X": data.scaler_X,
            "scaler_y": data.scaler_y,
            "feature_cols": data.feature_cols,
            "target_col": data.target_col,
        },
        artifact_dir / "preprocessing.joblib",
    )

    save_json(artifact_dir / "manual_fis_params.json", serializable_fis_params(manual_fis))
    save_json(artifact_dir / "ga_fis_params.json", serializable_fis_params(ga_fis))

    torch.save(
        {
            "n_age": N_AGE_MF,
            "n_bmi": N_BMI_MF,
            "n_smo": N_SMOKER_MF,
            "state_dict": ann_model.state_dict(),
        },
        artifact_dir / "ann_model.pt",
    )

    save_json(
        artifact_dir / "metrics.json",
        {
            "dataset_shape": list(result["df_shape"]),
            "metrics": result["metrics"],
            "notes": "Scores are generated from test split after training.",
        },
    )

    print("Training complete. Artifacts saved in ./artifacts")
    print("- preprocessing.joblib")
    print("- manual_fis_params.json")
    print("- ga_fis_params.json")
    print("- ann_model.pt")
    print("- metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models and save deployment artifacts")
    parser.add_argument("--force", action="store_true", help="Retrain even if artifacts already exist")
    args = parser.parse_args()
    main(force_retrain=args.force)
