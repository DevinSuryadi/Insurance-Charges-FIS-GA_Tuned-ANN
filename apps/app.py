import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from general.model_utils import NeuroFuzzyNet, SugenoFIS


st.set_page_config(page_title="Insurance Neuro-Fuzzy", page_icon="📊", layout="wide")

PROJECT_DIR = ROOT_DIR
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


@st.cache_resource
def load_artifacts():
    required = [
        ARTIFACT_DIR / "preprocessing.joblib",
        ARTIFACT_DIR / "manual_fis_params.json",
        ARTIFACT_DIR / "ga_fis_params.json",
        ARTIFACT_DIR / "ann_model.pt",
    ]

    missing = [str(p.name) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Artifact belum tersedia: "
            + ", ".join(missing)
            + ". Jalankan: python scripts/train_and_save_artifacts.py"
        )

    prep = joblib.load(ARTIFACT_DIR / "preprocessing.joblib")
    scaler_X = prep["scaler_X"]
    scaler_y = prep["scaler_y"]

    import json

    with open(ARTIFACT_DIR / "manual_fis_params.json", "r", encoding="utf-8") as f:
        manual_params = json.load(f)
    with open(ARTIFACT_DIR / "ga_fis_params.json", "r", encoding="utf-8") as f:
        ga_params = json.load(f)

    manual_fis = SugenoFIS(**manual_params)
    ga_fis = SugenoFIS(**ga_params)

    ann_payload = torch.load(ARTIFACT_DIR / "ann_model.pt", map_location="cpu")
    ann_model = NeuroFuzzyNet(
        n_age=ann_payload["n_age"],
        n_bmi=ann_payload["n_bmi"],
        n_smo=ann_payload["n_smo"],
    )
    ann_model.load_state_dict(ann_payload["state_dict"])
    ann_model.eval()

    metrics = None
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return scaler_X, scaler_y, manual_fis, ga_fis, ann_model, metrics


def denorm_charge(y_norm, scaler_y):
    return float(scaler_y.inverse_transform(np.array(y_norm).reshape(-1, 1)).ravel()[0])


def predict_all(age, bmi, smoker, scaler_X, scaler_y, manual_fis, ga_fis, ann_model):
    X_raw = np.array([[age, bmi, smoker]], dtype=float)
    X_scaled = scaler_X.transform(X_raw)

    y_manual = manual_fis.predict(X_scaled)[0]
    y_ga = ga_fis.predict(X_scaled)[0]
    with torch.no_grad():
        y_ann = float(ann_model(torch.FloatTensor(X_scaled)).item())

    return {
        "Manual FIS": denorm_charge(y_manual, scaler_y),
        "GA-Tuned FIS": denorm_charge(y_ga, scaler_y),
        "NeuroFuzzy ANN": denorm_charge(y_ann, scaler_y),
    }


def render_metrics(metrics_payload):
    st.subheader("Model Test Metrics")
    if not metrics_payload or "metrics" not in metrics_payload:
        st.info("metrics.json tidak ditemukan. Jalankan training script untuk menyimpan metrik test.")
        return

    rows = []
    mapping = {
        "manual": "Manual FIS",
        "ga": "GA-Tuned FIS",
        "ann": "NeuroFuzzy ANN",
    }

    for key, label in mapping.items():
        m = metrics_payload["metrics"].get(key, {})
        rows.append(
            {
                "Method": label,
                "R2 (norm)": round(m.get("R2_norm", float("nan")), 4),
                "RMSE (USD)": round(m.get("RMSE_usd", float("nan")), 2),
                "MAE (USD)": round(m.get("MAE_usd", float("nan")), 2),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    st.title("Prediksi Biaya Asuransi - Neuro-Fuzzy Deployment")
    st.caption("Model dari notebook dipisah menjadi artifact agar Streamlit ringan dan stabil.")

    try:
        scaler_X, scaler_y, manual_fis, ga_fis, ann_model, metrics = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.code("python train_and_save_artifacts.py")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        age = st.slider("Age", min_value=18, max_value=65, value=35)
        bmi = st.slider("BMI", min_value=15.0, max_value=55.0, value=30.0, step=0.1)
        smoker_label = st.selectbox("Smoker", options=["No", "Yes"], index=0)
        smoker = 1.0 if smoker_label == "Yes" else 0.0

        run_btn = st.button("Predict", type="primary")

    with col2:
        st.subheader("Prediction")
        if run_btn:
            preds = predict_all(age, bmi, smoker, scaler_X, scaler_y, manual_fis, ga_fis, ann_model)

            rows = [{"Method": k, "Predicted Charges (USD)": round(v, 2)} for k, v in preds.items()]
            pred_df = pd.DataFrame(rows).sort_values("Predicted Charges (USD)", ascending=False)

            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            st.bar_chart(pred_df.set_index("Method"))
        else:
            st.info("Klik Predict untuk melihat estimasi biaya asuransi.")

    st.divider()
    render_metrics(metrics)


if __name__ == "__main__":
    main()
