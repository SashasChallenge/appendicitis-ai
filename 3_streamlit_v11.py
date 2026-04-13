# python -m streamlit run "D:\人工智能\练手2\3_streamlit_v11.py"

# =========================
# 0) Page config（必须最先）
# =========================
import streamlit as st
st.set_page_config(
    page_title="Clinical Decision Support System",
    layout="wide",
    page_icon="🏥基于Logistic Regression的复杂性阑尾炎预测系统"
)

# =========================
# 1) Imports
# =========================
import os
import io
import pickle
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

plt.rcParams["font.sans-serif"] = ["microsoft yahei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 2) Helper functions
# =========================
def st_shap(plot, height=180):
    """Embed SHAP JS plot in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)

def get_risk_level(prob, low=0.30, high=0.70):
    """
    Return (level_name, color_hex, advice)
    """
    if prob < low:
        return "Low risk", "#2ECC71", "Routine follow-up is recommended."
    elif prob < high:
        return "Intermediate risk", "#F39C12", "Enhanced monitoring and further assessment are recommended."
    else:
        return "High risk", "#E74C3C", "Immediate clinical attention and intervention are recommended."

def fig_download_button(fig, filename, label="Download figure (PNG, 300 dpi)"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    st.download_button(label, buf, file_name=filename, mime="image/png")

def align_features(df, feature_names):
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return df[feature_names]

# =========================
# 3) Load artifacts (cached)
# =========================
@st.cache_resource
def load_artifacts(model_dir="saved_models"):
    model_path = os.path.join(model_dir, "Logistic_Regression.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    # scaler
    scaler = None
    for sp in [os.path.join(model_dir, "scaler1.pkl"), os.path.join(model_dir, "scaler.pkl")]:
        if os.path.exists(sp):
            try:
                scaler = joblib.load(sp)
            except Exception:
                with open(sp, "rb") as f:
                    scaler = pickle.load(f)
            break
    if scaler is None:
        raise FileNotFoundError("Scaler not found: scaler1.pkl or scaler.pkl")

    # feature names
    feature_names = None
    for fp in [os.path.join(model_dir, "feature_names1.pkl"), os.path.join(model_dir, "feature_names.pkl")]:
        if os.path.exists(fp):
            try:
                feature_names = joblib.load(fp)
            except Exception:
                with open(fp, "rb") as f:
                    feature_names = pickle.load(f)
            break
    if feature_names is None:
        raise FileNotFoundError("Feature names not found: feature_names1.pkl or feature_names.pkl")

    return model, scaler, list(feature_names), model_path

try:
    model, scaler, feature_names, model_path = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load artifacts: {e}")
    st.stop()

# =========================
# 4) SHAP background data + explainer (cached)
# =========================
@st.cache_resource
def load_background_data(feature_names, _scaler,
                         candidate_files=("train_data.csv", "external_validation_data.csv"),
                         max_bg=500):
    """
    Load background data for SHAP; return scaled numpy array or None.
    Robust version: never references variables before assignment.
    """
    for f in candidate_files:
        if not os.path.exists(f):
            continue

        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        # 去掉标签列（如果存在）
        if "Target" in df.columns:
            df = df.drop(columns=["Target"])

        # 去掉非数值列
        non_num = df.select_dtypes(include=["object"]).columns.tolist()
        if non_num:
            df = df.drop(columns=non_num)

        # 特征对齐（这一步可能抛异常）
        try:
            X_df = align_features(df, feature_names)  # DataFrame
        except Exception:
            continue

        # 抽样（DataFrame 用 iloc）
        if len(X_df) > max_bg:
            np.random.seed(42)
            idx = np.random.choice(len(X_df), max_bg, replace=False)
            X_df = X_df.iloc[idx]

        # 用 DataFrame transform，避免“无特征名”警告
        try:
            X_scaled = _scaler.transform(X_df)
        except Exception:
            # 万一 scaler 只接受 ndarray
            X_scaled = _scaler.transform(X_df.values)

        return np.asarray(X_scaled)

    return None


@st.cache_resource
def build_shap_explainer(_model, X_bg_scaled, feature_names):
    """
    Prefer LinearExplainer for LogisticRegression; fallback to generic Explainer.
    """
    if X_bg_scaled is None:
        return None, "No background data (SHAP disabled)"

    # LogisticRegression -> LinearExplainer (recommended)
    try:
        expl = shap.LinearExplainer(_model, X_bg_scaled, feature_names=feature_names)
        return expl, "LinearExplainer"
    except Exception:
        masker = shap.maskers.Independent(X_bg_scaled)
        expl = shap.Explainer(lambda x: _model.predict_proba(x)[:, 1], masker, feature_names=feature_names)
        return expl, "Explainer(Independent masker)"

def compute_shap_one(explainer, x_scaled_2d):
    """
    Return (shap_values_1d, base_value_float)
    Compatible with LinearExplainer and shap.Explainer.
    """
    # New-style call
    try:
        exp = explainer(x_scaled_2d)
        sv = exp.values
        bv = exp.base_values
        sv_1d = sv[0] if sv.ndim == 2 else sv
        bv0 = float(np.ravel(bv)[0])
        return np.asarray(sv_1d), bv0
    except Exception:
        pass

    # Old-style (just in case)
    sv_raw = explainer.shap_values(x_scaled_2d)
    if isinstance(sv_raw, list):
        sv_1d = sv_raw[1][0]
    else:
        sv_1d = sv_raw[0]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1] if len(base) > 1 else base[0]
    return np.asarray(sv_1d), float(np.ravel(base)[0])

X_bg_scaled = load_background_data(feature_names, scaler)
explainer, explainer_type = build_shap_explainer(model, X_bg_scaled, feature_names)

# =========================
# 5) Header
# =========================
st.title("🏥 A System for Predicting Complicated Appendicitis (Logistic Regression)")

m1, m2, m3 = st.columns(3)
m1.metric("Model", os.path.basename(model_path))
m2.metric("Features", len(feature_names))
m3.metric("SHAP explainer", explainer_type)

with st.expander("Notes (recommended for reproducibility)", expanded=False):
    st.markdown(
        """
- Inputs are **standardized** using the saved scaler (transform only).
- Predicted probability refers to the **positive class** (Complicated appendicitis = 1).
- SHAP explanations are computed on the model input space (standardized features).
        """
    )

# sidebar: thresholds
st.sidebar.header("Settings")
thr_pred = st.sidebar.slider("Classification threshold", 0.05, 0.95, 0.50, 0.01)
thr_low = st.sidebar.slider("Low-risk cutoff", 0.05, 0.95, 0.30, 0.01)
thr_high = st.sidebar.slider("High-risk cutoff", 0.05, 0.95, 0.70, 0.01)
if thr_low >= thr_high:
    st.sidebar.error("Low-risk cutoff must be smaller than high-risk cutoff.")

tab1, tab2 = st.tabs(["📝 单例预测 (手动输入)", "📂 批量预测 (上传Excel/CSV)"])

# =========================
# 6) Tab1: single prediction (NOT EMPTY)
# =========================
with tab1:
    st.info("适用于对单个样本进行快速风险评估和归因分析（SHAP）。")

    with st.form("single_predict_form"):
        inputs = {}
        n_cols = 4 if len(feature_names) > 10 else 2
        cols = st.columns(n_cols)

        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                inputs[feat] = st.number_input(feat, value=0.0, format="%.4f")

        submitted = st.form_submit_button("🚀 开始预测")

    if submitted:
        try:
            x_raw_df = pd.DataFrame([inputs], columns=feature_names)
            x_scaled = scaler.transform(x_raw_df)  # 保留列名

            prob = float(model.predict_proba(x_scaled)[0, 1])
            pred = int(prob >= thr_pred)

            level, color, advice = get_risk_level(prob, low=thr_low, high=thr_high)

            st.divider()
            c1, c2, c3 = st.columns([1, 1, 2])
            c1.metric("Predicted probability", f"{prob:.4f}")
            c2.metric("Predicted class", "Positive (1)" if pred == 1 else "Negative (0)")
            c3.markdown(
                f"<div style='background:{color}15; border-left:6px solid {color}; "
                f"padding:12px; border-radius:6px;'>"
                f"<b>Risk level:</b> <span style='color:{color}; font-weight:700'>{level}</span><br>"
                f"<b>Recommendation:</b> {advice}</div>",
                unsafe_allow_html=True
            )
            st.progress(min(prob, 1.0))

            # SHAP explanation
            st.divider()
            st.subheader("🔍 SHAP interpretability")

            if explainer is None:
                st.warning(
                    "SHAP is disabled because background data was not found. "
                    "Place `train_data.csv` or `external_validation_data.csv` in the project root to enable SHAP."
                )
            else:
                with st.spinner("Computing SHAP..."):
                    sv_1d, bv = compute_shap_one(explainer, x_scaled)

                # Waterfall
                st.markdown("**1) Waterfall plot**")
                exp_obj = shap.Explanation(
                    values=sv_1d,
                    base_values=bv,
                    data=x_raw_df.iloc[0].values,   # show raw values in labels
                    feature_names=feature_names
                )

                fig = plt.figure(figsize=(11, 6.5))
                shap.plots.waterfall(exp_obj, max_display=12, show=False)
                st.pyplot(fig, use_container_width=True)
                fig_download_button(fig, "single_shap_waterfall.png")
                plt.close(fig)

                # Force plot (interactive)
                st.markdown("**2) Force plot (interactive)**")
                try:
                    force = shap.force_plot(
                        bv, sv_1d, x_raw_df.iloc[0],
                        feature_names=feature_names,
                        matplotlib=False
                    )
                    st_shap(force, height=190)
                except Exception as e:
                    st.info(f"Interactive force plot is not available: {e}")

                # Contribution table
                st.markdown("**3) Contribution ranking**")
                df_contrib = pd.DataFrame({
                    "Feature": feature_names,
                    "Value(raw)": x_raw_df.iloc[0].values,
                    "SHAP": sv_1d,
                    "|SHAP|": np.abs(sv_1d),
                    "Direction": np.where(sv_1d >= 0, "↑ increases risk", "↓ decreases risk")
                }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)
                df_contrib.index = df_contrib.index + 1
                st.dataframe(df_contrib, use_container_width=True, height=420)

        except Exception as e:
            st.error(f"运行出错: {e}")
            import traceback
            st.text(traceback.format_exc())

# =========================
# 7) Tab2: batch prediction (NOT EMPTY)
# =========================
with tab2:
    st.info("适用于处理多条数据。上传 Excel (.xlsx) 或 CSV，并自动生成预测结果与可选 SHAP 解释。")

    with st.expander("📥 下载数据模板", expanded=False):
        template_df = pd.DataFrame(columns=["Patient_ID"] + feature_names)
        st.download_button(
            "下载 CSV 模板",
            template_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="prediction_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("上传文件", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.success(f"✅ 成功读取 {len(df_upload)} 条数据。")

            # Align features and scale
            X_batch = align_features(df_upload, feature_names)  # DataFrame
            X_batch_scaled = scaler.transform(X_batch)  # 直接传 DataFrame

            probs = model.predict_proba(X_batch_scaled)[:, 1]
            preds = (probs >= thr_pred).astype(int)

            df_result = df_upload.copy()
            df_result["Pred_Prob"] = np.round(probs, 6)
            df_result["Pred_Class"] = preds
            df_result["Risk_Level"] = [get_risk_level(p, thr_low, thr_high)[0] for p in probs]

            st.divider()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("N", len(df_result))
            s2.metric("Predicted positive", int(preds.sum()))
            s3.metric("Predicted negative", int((1 - preds).sum()))
            s4.metric("Mean probability", f"{probs.mean():.4f}")

            st.markdown("**预测结果表**")
            st.dataframe(df_result, use_container_width=True, height=420)

            st.download_button(
                "💾 下载预测结果 (.csv)",
                df_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="prediction_results.csv",
                mime="text/csv"
            )

            # Optional: explain a selected sample
            st.divider()
            st.subheader("🔍 解释某条样本（SHAP）")

            idx = st.selectbox(
                "选择要解释的行号 (index)",
                options=list(df_result.index),
                format_func=lambda i: f"行 {i} | Prob={df_result.loc[i,'Pred_Prob']:.4f} | {df_result.loc[i,'Risk_Level']}"
            )

            if st.button("解释该样本", type="primary"):
                if explainer is None:
                    st.warning("SHAP is disabled (background data not found).")
                else:
                    x_single_raw = X_batch.iloc[[idx]]
                    x_single_scaled = X_batch_scaled[idx].reshape(1, -1)

                    with st.spinner("Computing SHAP..."):
                        sv_1d, bv = compute_shap_one(explainer, x_single_scaled)

                    exp_obj = shap.Explanation(
                        values=sv_1d,
                        base_values=bv,
                        data=x_single_raw.iloc[0].values,
                        feature_names=feature_names
                    )

                    fig = plt.figure(figsize=(11, 6.5))
                    shap.plots.waterfall(exp_obj, max_display=12, show=False)
                    st.pyplot(fig, use_container_width=True)
                    fig_download_button(fig, f"batch_row_{idx}_shap_waterfall.png")
                    plt.close(fig)

                    try:
                        force = shap.force_plot(
                            bv, sv_1d, x_single_raw.iloc[0],
                            feature_names=feature_names,
                            matplotlib=False
                        )
                        st_shap(force, height=190)
                    except Exception as e:
                        st.info(f"Interactive force plot not available: {e}")

        except Exception as e:
            st.error(f"处理文件时发生错误: {e}")
            import traceback
            st.text(traceback.format_exc())

# =========================
# Footer
# =========================
st.divider()
st.markdown(
    "<div style='text-align:center; color:#999; font-size:12px;'>"
    "⚠️ Research use only. Not intended for clinical diagnosis or treatment decisions."
    "</div>",
    unsafe_allow_html=True
)
