import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from general.model_utils import NeuroFuzzyNet, SugenoFIS, load_dataset


st.set_page_config(page_title="Insurance Neuro-Fuzzy", page_icon=":bar_chart:", layout="wide")

PROJECT_DIR = ROOT_DIR
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
DATASET_PATH = PROJECT_DIR / "insurance.csv"
STYLE_PATH = PROJECT_DIR / "apps" / "style" / "style.css"


def apply_custom_style():
    if STYLE_PATH.exists():
        with open(STYLE_PATH, "r", encoding="utf-8") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


def render_section_marker(title, subtitle=""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def render_section_divider():
    st.divider()


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


@st.cache_data
def load_data_for_eda():
    return load_dataset(csv_path=str(DATASET_PATH)).copy()


def denorm_array(y_norm_array, scaler_y):
    return scaler_y.inverse_transform(np.array(y_norm_array).reshape(-1, 1)).ravel()


def predict_models_on_frame(features_df, scaler_X, scaler_y, manual_fis, ga_fis, ann_model):
    X_raw = features_df[["age", "bmi", "smoker"]].astype(float).to_numpy()
    X_scaled = scaler_X.transform(X_raw)

    y_manual_norm = manual_fis.predict(X_scaled)
    y_ga_norm = ga_fis.predict(X_scaled)
    with torch.no_grad():
        y_ann_norm = ann_model(torch.FloatTensor(X_scaled)).detach().cpu().numpy().ravel()

    return pd.DataFrame(
        {
            "Manual FIS": denorm_array(y_manual_norm, scaler_y),
            "GA-Tuned FIS": denorm_array(y_ga_norm, scaler_y),
            "NeuroFuzzy ANN": denorm_array(y_ann_norm, scaler_y),
        }
    )


def get_metrics_frame(metrics_payload):
    if not metrics_payload or "metrics" not in metrics_payload:
        return None

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

    return pd.DataFrame(rows)


def chart_pred_bar(pred_df):
    chart_df = pred_df.rename(columns={"Predicted Charges (USD)": "Predicted Charges"})
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Method:N", sort="-y", title="Model"),
            y=alt.Y("Predicted Charges:Q", title="Predicted Charges (USD)"),
            color=alt.Color("Method:N", legend=None),
            tooltip=["Method", alt.Tooltip("Predicted Charges:Q", format=",.2f")],
        )
        .properties(height=320)
    )


def chart_grouped_metrics(metrics_df):
    chart_df = metrics_df.melt(
        id_vars=["Method"],
        value_vars=["R2 (norm)", "RMSE (USD)", "MAE (USD)"],
        var_name="Metric",
        value_name="Value",
    )
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metric:N", title="Metric"),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Method:N", title="Method"),
            column=alt.Column("Method:N", title=None),
            tooltip=["Method", "Metric", alt.Tooltip("Value:Q", format=",.4f")],
        )
        .properties(height=280)
    )


def chart_error_metrics(metrics_df):
    chart_df = metrics_df.melt(
        id_vars=["Method"],
        value_vars=["RMSE (USD)", "MAE (USD)"],
        var_name="Metric",
        value_name="Value",
    )
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Method:N", title="Model"),
            y=alt.Y("Value:Q", title="Error (USD)"),
            color=alt.Color("Metric:N", title="Metric"),
            xOffset="Metric:N",
            tooltip=["Method", "Metric", alt.Tooltip("Value:Q", format=",.2f")],
        )
        .properties(height=320)
    )


def chart_distribution(df, col_name, title, bins=30):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{col_name}:Q", bin=alt.Bin(maxbins=bins), title=title),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=240)
    )


def compute_batch_error_table(batch_df):
    actual = batch_df["charges"].to_numpy()
    rows = []
    for model_col in ["Manual FIS", "GA-Tuned FIS", "NeuroFuzzy ANN"]:
        pred = batch_df[model_col].to_numpy()
        abs_err = np.abs(actual - pred)
        sq_err = (actual - pred) ** 2
        sst = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1.0 - (np.sum(sq_err) / sst if sst > 0 else np.nan)
        rows.append(
            {
                "Method": model_col,
                "MAE (USD)": float(np.mean(abs_err)),
                "RMSE (USD)": float(np.sqrt(np.mean(sq_err))),
                "R2": float(r2),
            }
        )
    return pd.DataFrame(rows)


def render_tab_perbandingan(metrics_payload, scaler_X, scaler_y, manual_fis, ga_fis, ann_model):
    render_section_marker(
        "Perbandingan Model",
        "Halaman utama untuk membandingkan prediksi, metrik, dan simulasi ketiga pendekatan.",
    )

    render_section_marker("Perbandingan Prediksi pada Profil Input")
    input_col1, input_col2, input_col3 = st.columns(3)
    age = input_col1.slider("Age", min_value=18, max_value=65, value=35)
    bmi = input_col2.slider("BMI", min_value=15.0, max_value=55.0, value=30.0, step=0.1)
    smoker_label = input_col3.selectbox("Smoker", options=["No", "Yes"], index=0)
    smoker = 1.0 if smoker_label == "Yes" else 0.0

    feature_df = pd.DataFrame(
        {"age": [float(age)], "bmi": [float(bmi)], "smoker": [float(smoker)]}
    )
    pred_raw = predict_models_on_frame(feature_df, scaler_X, scaler_y, manual_fis, ga_fis, ann_model).iloc[0].to_dict()
    pred_rows = [
        {"Method": key, "Predicted Charges (USD)": round(float(value), 2)}
        for key, value in pred_raw.items()
    ]
    pred_df = pd.DataFrame(pred_rows).sort_values("Predicted Charges (USD)", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Manual FIS", f"${pred_raw['Manual FIS']:,.2f}")
    c2.metric("GA-Tuned FIS", f"${pred_raw['GA-Tuned FIS']:,.2f}")
    c3.metric("NeuroFuzzy ANN", f"${pred_raw['NeuroFuzzy ANN']:,.2f}")

    spread = float(pred_df["Predicted Charges (USD)"].max() - pred_df["Predicted Charges (USD)"].min())
    st.caption(f"Spread antar model: ${spread:,.2f}")
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    st.altair_chart(chart_pred_bar(pred_df), use_container_width=True)

    render_section_divider()

    metrics_df = get_metrics_frame(metrics_payload)
    if metrics_df is None:
        st.warning("metrics.json belum tersedia. Jalankan: python scripts/train_and_save_artifacts.py")
    else:
        render_section_marker("Metrik Test")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.altair_chart(chart_error_metrics(metrics_df), use_container_width=True)

    render_section_divider()

    render_section_marker("Simulasi Kurva Prediksi")
    st.caption("Lihat bagaimana prediksi berubah terhadap rentang Age atau BMI. Nilai tetap mengikuti profil input di atas.")

    sweep_feature = st.selectbox("Sumbu simulasi", options=["Age", "BMI"], index=0)
    fixed_age = age
    fixed_bmi = bmi
    fixed_smoker = smoker

    if sweep_feature == "Age":
        sweep_vals = np.arange(18, 66, 1)
        sim_df = pd.DataFrame(
            {
                "age": sweep_vals.astype(float),
                "bmi": np.full_like(sweep_vals, fixed_bmi, dtype=float),
                "smoker": np.full_like(sweep_vals, fixed_smoker, dtype=float),
            }
        )
        x_title = "Age"
        x_col = "age"
    else:
        sweep_vals = np.round(np.arange(15.0, 55.1, 0.5), 1)
        sim_df = pd.DataFrame(
            {
                "age": np.full_like(sweep_vals, fixed_age, dtype=float),
                "bmi": sweep_vals.astype(float),
                "smoker": np.full_like(sweep_vals, fixed_smoker, dtype=float),
            }
        )
        x_title = "BMI"
        x_col = "bmi"

    pred_curve = predict_models_on_frame(sim_df, scaler_X, scaler_y, manual_fis, ga_fis, ann_model)
    curve_df = pd.concat([sim_df[[x_col]].reset_index(drop=True), pred_curve], axis=1)
    curve_long = curve_df.melt(id_vars=[x_col], var_name="Method", value_name="Predicted Charges")

    curve_chart = (
        alt.Chart(curve_long)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_title),
            y=alt.Y("Predicted Charges:Q", title="Predicted Charges (USD)"),
            color=alt.Color("Method:N", title="Model"),
            tooltip=[x_col, "Method", alt.Tooltip("Predicted Charges:Q", format=",.2f")],
        )
        .properties(height=380)
    )
    st.altair_chart(curve_chart, use_container_width=True)


def render_tab_eda(df):
    render_section_marker(
        "EDA Dataset",
        "Eksplorasi data insurance: statistik dasar, distribusi, dan relasi fitur terhadap charges.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Mean Charges", f"${df['charges'].mean():,.2f}")
    c4.metric("Smoker Ratio", f"{(df['smoker'].mean() * 100):.2f}%")

    with st.expander("Lihat sample data dan statistik", expanded=False):
        st.dataframe(df.head(15), use_container_width=True)
        st.markdown("**Descriptive statistics**")
        st.dataframe(df.describe().T, use_container_width=True)
        missing_df = pd.DataFrame({"Missing Count": df.isna().sum()})
        st.markdown("**Missing values**")
        st.dataframe(missing_df, use_container_width=True)

    render_section_divider()
    render_section_marker("Distribusi Fitur")
    dist_c1, dist_c2, dist_c3 = st.columns(3)
    dist_c1.altair_chart(chart_distribution(df, "age", "Age", bins=20), use_container_width=True)
    dist_c2.altair_chart(chart_distribution(df, "bmi", "BMI", bins=20), use_container_width=True)
    dist_c3.altair_chart(chart_distribution(df, "charges", "Charges", bins=30), use_container_width=True)

    render_section_divider()
    render_section_marker("Relasi Fitur vs Charges")
    scatter_df = df.copy()
    scatter_df["smoker_label"] = np.where(scatter_df["smoker"] >= 0.5, "Smoker", "Non-Smoker")

    sc1, sc2 = st.columns(2)

    age_scatter = (
        alt.Chart(scatter_df)
        .mark_circle(size=45, opacity=0.65)
        .encode(
            x=alt.X("age:Q", title="Age"),
            y=alt.Y("charges:Q", title="Charges (USD)"),
            color=alt.Color("smoker_label:N", title="Category"),
            tooltip=["age", "bmi", "smoker_label", alt.Tooltip("charges:Q", format=",.2f")],
        )
        .properties(height=330, title="Age vs Charges")
    )
    sc1.altair_chart(age_scatter, use_container_width=True)

    bmi_scatter = (
        alt.Chart(scatter_df)
        .mark_circle(size=45, opacity=0.65)
        .encode(
            x=alt.X("bmi:Q", title="BMI"),
            y=alt.Y("charges:Q", title="Charges (USD)"),
            color=alt.Color("smoker_label:N", title="Category"),
            tooltip=["age", "bmi", "smoker_label", alt.Tooltip("charges:Q", format=",.2f")],
        )
        .properties(height=330, title="BMI vs Charges")
    )
    sc2.altair_chart(bmi_scatter, use_container_width=True)

    render_section_divider()
    render_section_marker("Agregasi Charges berdasarkan Status Smoker")
    smoker_summary = (
        scatter_df.groupby("smoker_label", as_index=False)
        .agg(
            count=("charges", "count"),
            mean_charges=("charges", "mean"),
            median_charges=("charges", "median"),
        )
        .sort_values("mean_charges", ascending=False)
    )
    st.dataframe(smoker_summary, use_container_width=True, hide_index=True)

    smoker_bar = (
        alt.Chart(smoker_summary)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("smoker_label:N", title="Category"),
            y=alt.Y("mean_charges:Q", title="Average Charges (USD)"),
            color=alt.Color("smoker_label:N", legend=None),
            tooltip=["smoker_label", alt.Tooltip("mean_charges:Q", format=",.2f"), "count"],
        )
        .properties(height=280)
    )
    st.altair_chart(smoker_bar, use_container_width=True)


def render_tab_batch_eval(df, scaler_X, scaler_y, manual_fis, ga_fis, ann_model):
    render_section_marker(
        "Evaluasi Batch dan Insight",
        "Bandingkan prediksi ketiga model terhadap nilai actual pada sampel dataset.",
    )

    sample_size = st.slider(
        "Jumlah data evaluasi",
        min_value=100,
        max_value=len(df),
        value=min(400, len(df)),
        step=50,
    )

    eval_btn = st.button("Run Batch Evaluation")
    if eval_btn:
        eval_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        model_preds = predict_models_on_frame(eval_df[["age", "bmi", "smoker"]], scaler_X, scaler_y, manual_fis, ga_fis, ann_model)
        st.session_state["batch_eval_df"] = pd.concat([eval_df, model_preds], axis=1)

    if "batch_eval_df" not in st.session_state:
        st.info("Klik Run Batch Evaluation untuk menampilkan hasil evaluasi batch.")
        return

    batch_df = st.session_state["batch_eval_df"]
    err_df = compute_batch_error_table(batch_df)
    render_section_divider()
    render_section_marker("Ringkasan Error")
    st.dataframe(err_df.round(4), use_container_width=True, hide_index=True)

    err_chart_df = err_df.melt(id_vars=["Method"], value_vars=["MAE (USD)", "RMSE (USD)"], var_name="Metric", value_name="Value")
    err_chart = (
        alt.Chart(err_chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Method:N", title="Model"),
            y=alt.Y("Value:Q", title="Error (USD)"),
            color=alt.Color("Metric:N", title="Metric"),
            xOffset="Metric:N",
            tooltip=["Method", "Metric", alt.Tooltip("Value:Q", format=",.2f")],
        )
        .properties(height=320)
    )
    st.altair_chart(err_chart, use_container_width=True)

    render_section_divider()
    render_section_marker("Actual vs Predicted")
    selected_model = st.selectbox(
        "Pilih model",
        options=["Manual FIS", "GA-Tuned FIS", "NeuroFuzzy ANN"],
        index=2,
    )

    scatter_df = pd.DataFrame(
        {
            "Actual": batch_df["charges"].astype(float),
            "Predicted": batch_df[selected_model].astype(float),
        }
    )

    min_val = float(min(scatter_df["Actual"].min(), scatter_df["Predicted"].min()))
    max_val = float(max(scatter_df["Actual"].max(), scatter_df["Predicted"].max()))
    line_df = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})

    points = (
        alt.Chart(scatter_df)
        .mark_circle(size=45, opacity=0.55)
        .encode(
            x=alt.X("Actual:Q", title="Actual Charges (USD)"),
            y=alt.Y("Predicted:Q", title="Predicted Charges (USD)"),
            tooltip=[alt.Tooltip("Actual:Q", format=",.2f"), alt.Tooltip("Predicted:Q", format=",.2f")],
        )
    )
    diagonal = alt.Chart(line_df).mark_line(strokeDash=[8, 6], color="#111111").encode(x="x:Q", y="y:Q")
    st.altair_chart((points + diagonal).properties(height=380), use_container_width=True)

    render_section_divider()
    render_section_marker("Sampel Hasil Prediksi")
    preview_cols = ["age", "bmi", "smoker", "charges", "Manual FIS", "GA-Tuned FIS", "NeuroFuzzy ANN"]
    st.dataframe(batch_df[preview_cols].head(20).round(2), use_container_width=True)


def main():
    apply_custom_style()
    st.title("Insurance Charges Prediction Dashboard")
    st.caption("Deployment model Manual FIS, GA-Tuned FIS, dan NeuroFuzzy ANN dengan navigasi tab.")

    try:
        scaler_X, scaler_y, manual_fis, ga_fis, ann_model, metrics = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.code("python scripts/train_and_save_artifacts.py")
        st.stop()

    df = load_data_for_eda()

    tab1, tab2, tab3 = st.tabs(
        [
            "Perbandingan Model",
            "EDA Dataset",
            "Evaluasi Batch",
        ]
    )

    with tab1:
        render_tab_perbandingan(metrics, scaler_X, scaler_y, manual_fis, ga_fis, ann_model)

    with tab2:
        render_tab_eda(df)

    with tab3:
        render_tab_batch_eval(df, scaler_X, scaler_y, manual_fis, ga_fis, ann_model)


if __name__ == "__main__":
    main()
