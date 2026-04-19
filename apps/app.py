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
STYLE_PATH = PROJECT_DIR / "apps" / "Style" / "style.css"


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


def apply_chart_theme(chart):
    return (
        chart.configure(background="white")
        .configure_view(fill="white", stroke="#e5e7eb")
        .configure_axis(
            labelColor="#111827",
            titleColor="#111827",
            gridColor="#e5e7eb",
            domainColor="#d1d5db",
            tickColor="#d1d5db",
        )
        .configure_legend(labelColor="#111827", titleColor="#111827")
        .configure_title(color="#111827")
    )


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


def chart_distribution(df, col_name, title, bins=30):
    return apply_chart_theme(
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{col_name}:Q", bin=alt.Bin(maxbins=bins), title=title),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=240)
    )


def render_tab_perbandingan(df, scaler_X, scaler_y, manual_fis, ga_fis, ann_model):
    render_section_marker(
        "Perbandingan Model",
        "Halaman utama untuk membandingkan hasil prediksi dan simulasi ketiga pendekatan.",
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

    st.markdown("<div class='input-result-gap'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Predicted Charges (USD)")
        st.metric("Manual FIS", f"${pred_raw['Manual FIS']:,.2f}")
    with c2:
        st.caption("Predicted Charges (USD)")
        st.metric("GA-Tuned FIS", f"${pred_raw['GA-Tuned FIS']:,.2f}")
    with c3:
        st.caption("Predicted Charges (USD)")
        st.metric("NeuroFuzzy ANN", f"${pred_raw['NeuroFuzzy ANN']:,.2f}")

    avg_pred = float(np.mean(list(pred_raw.values())))
    st.caption(f"Rata-rata estimasi charges: ${avg_pred:,.2f}")

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

    curve_chart = apply_chart_theme(
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

    render_section_divider()
    render_section_marker("Actual vs Predicted (Sejajar)")
    st.caption("Perbandingan prediksi terhadap nilai actual untuk ketiga model pada seluruh dataset.")

    eval_df = df[["age", "bmi", "smoker", "charges"]].reset_index(drop=True)
    eval_preds = predict_models_on_frame(
        eval_df[["age", "bmi", "smoker"]],
        scaler_X,
        scaler_y,
        manual_fis,
        ga_fis,
        ann_model,
    )
    eval_df = pd.concat([eval_df, eval_preds], axis=1)

    model_cols = ["Manual FIS", "GA-Tuned FIS", "NeuroFuzzy ANN"]
    min_val = float(min(eval_df["charges"].min(), eval_df[model_cols].min().min()))
    max_val = float(max(eval_df["charges"].max(), eval_df[model_cols].max().max()))
    line_df = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})
    x_domain = [min_val, max_val]
    y_domain = [min_val, max_val]

    plot_cols = st.columns(3)
    for col, model_name in zip(plot_cols, model_cols):
        scatter_df = pd.DataFrame(
            {
                "Actual": eval_df["charges"].astype(float),
                "Predicted": eval_df[model_name].astype(float),
            }
        )
        points = (
            alt.Chart(scatter_df)
            .mark_circle(size=34, opacity=0.5)
            .encode(
                x=alt.X("Actual:Q", title="Actual", scale=alt.Scale(domain=x_domain)),
                y=alt.Y("Predicted:Q", title="Predicted", scale=alt.Scale(domain=y_domain)),
                tooltip=[
                    alt.Tooltip("Actual:Q", format=",.2f"),
                    alt.Tooltip("Predicted:Q", format=",.2f"),
                ],
            )
        )
        diagonal = alt.Chart(line_df).mark_line(strokeDash=[8, 6], color="#111111").encode(x="x:Q", y="y:Q")
        chart = apply_chart_theme((points + diagonal).properties(height=320, title=model_name))
        col.altair_chart(chart, use_container_width=True)


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

    age_scatter = apply_chart_theme(
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

    bmi_scatter = apply_chart_theme(
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

    smoker_bar = apply_chart_theme(
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


def main():
    apply_custom_style()
    st.title("Insurance Charges Prediction Dashboard")
    st.caption("Deployment model Manual FIS, GA-Tuned FIS, dan NeuroFuzzy ANN dengan navigasi tab.")

    try:
        scaler_X, scaler_y, manual_fis, ga_fis, ann_model, _metrics = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.code("python scripts/train_and_save_artifacts.py")
        st.stop()

    df = load_data_for_eda()

    tab1, tab2 = st.tabs(
        [
            "Perbandingan Model",
            "EDA Dataset",
        ]
    )

    with tab1:
        render_tab_perbandingan(df, scaler_X, scaler_y, manual_fis, ga_fis, ann_model)

    with tab2:
        render_tab_eda(df)


if __name__ == "__main__":
    main()
