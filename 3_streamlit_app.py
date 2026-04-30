# =====================================================
# 复杂性阑尾炎预测系统 (PWA + Streamlit)
# 启动命令: streamlit run 3_streamlit_app.py
# =====================================================

import streamlit as st

# 0) Page config（必须最先）
st.set_page_config(
    page_title="Complicated Appendicitis Prediction",
    layout="wide",
    page_icon="🏥"
)

# PWA 注入（使网页可添加到手机桌面）
import streamlit.components.v1 as components

def inject_pwa():
    pwa_html = """
    <link rel="manifest" href="/app/static/manifest.json">
    <meta name="theme-color" content="#FF4B4B">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="阑尾炎AI">
    <link rel="apple-touch-icon" href="/app/static/icon-192.png">
    <script>
    if('serviceWorker' in navigator){
      navigator.serviceWorker.register('/app/static/service-worker.js')
        .then(r=>console.log('SW ok',r.scope))
        .catch(e=>console.log('SW fail',e));
    }
    </script>
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    @media(max-width:768px){
      .block-container{padding-left:0.8rem;padding-right:0.8rem;}
      h1{font-size:1.4rem!important;}
    }
    </style>
    """
    components.html(pwa_html, height=0)

inject_pwa()

# 1) Imports
import os, io, pickle, joblib, warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["microsoft yahei"]
plt.rcParams["axes.unicode_minus"] = False

# =====================================================
# ✅ 2) Feature labels with units (DISPLAY ONLY)
# 只用于界面/图形展示，不改变模型输入列名
# =====================================================
feature_label_map = {
    "Age": "Age (years)",
    "WBC": "WBC (×10⁹/L)",
    "RBC": "RBC (×10¹²/L)",
    "PLT": "PLT (×10⁹/L)",
    "GR%": "GR% (%)",
    "LC": "LC (×10⁹/L)",
    "MONO": "MONO (×10⁹/L)",
    "MONO%": "MONO% (%)",

    # 注意：PCT如果是降钙素原通常 ng/mL；若是血小板压积则为 %
    "PCT": "PCT (ng/mL)",

    "NLR": "NLR (unitless)",
    "PLR": "PLR (unitless)",

    "Alb": "Albumin (g/L)",
    "Albumin": "Albumin (g/L)",
    "Total protein": "Total protein (g/L)",

    # 按你真实单位修改：g/L 或 mg/L
    "Prealbumin": "Prealbumin (g/L)",

    "Urea": "Urea (mmol/L)",
    "Creatinine": "Creatinine (μmol/L)",
    "LDH": "LDH (U/L)",
    "K": "K⁺ (mmol/L)",
    "PT": "PT (s)",

    # 按你真实单位修改：mg/L FEU 或 μg/mL FEU
    "D_dimer": "D-dimer (mg/L FEU)",
    "D-dimer": "D-dimer (mg/L FEU)",
}

def get_display_label(feat: str) -> str:
    return feature_label_map.get(feat, feat)

# 3) Helper functions
def st_shap(plot, height=180):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)

def get_risk_level(prob, low=0.30, high=0.70):
    if prob < low:
        return "Low risk", "🟢", "#2ECC71", "Routine follow-up is recommended."
    elif prob < high:
        return "Intermediate risk", "🟡", "#F39C12", "Enhanced monitoring and further assessment are recommended."
    else:
        return "High risk", "🔴", "#E74C3C", "Immediate clinical attention and intervention are recommended."

def fig_download(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    st.download_button("📥 Download figure", buf, file_name=filename, mime="image/png")

def align_features(df, feature_names):
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[feature_names]

# 4) Load model (cached)
@st.cache_resource
def load_artifacts(model_dir="saved_models"):
    model_path = os.path.join(model_dir, "Logistic_Regression.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    scaler = None
    for s in ["scaler1.pkl", "scaler.pkl"]:
        sp = os.path.join(model_dir, s)
        if os.path.exists(sp):
            scaler = joblib.load(sp)
            break
    if scaler is None:
        raise FileNotFoundError("Scaler not found")

    feature_names = None
    for f in ["feature_names1.pkl", "feature_names.pkl"]:
        fp = os.path.join(model_dir, f)
        if os.path.exists(fp):
            feature_names = joblib.load(fp)
            break
    if feature_names is None:
        raise FileNotFoundError("Feature names not found")

    return model, scaler, list(feature_names), model_path

try:
    model, scaler, feature_names, model_path = load_artifacts()
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# 让“显示名列表”与 feature_names 一一对应
display_feature_names = [get_display_label(f) for f in feature_names]

# 5) SHAP explainer (cached)
@st.cache_resource
def load_bg_data(feature_names, _scaler,
                 files=("train_data.csv", "external_validation_data.csv"),
                 max_bg=500):
    for f in files:
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "Target" in df.columns:
            df = df.drop(columns=["Target"])
        non_num = df.select_dtypes(include=["object"]).columns.tolist()
        if non_num:
            df = df.drop(columns=non_num)
        try:
            X_df = align_features(df, feature_names)
        except Exception:
            continue
        if len(X_df) > max_bg:
            np.random.seed(42)
            X_df = X_df.iloc[np.random.choice(len(X_df), max_bg, replace=False)]
        try:
            return np.asarray(_scaler.transform(X_df))
        except Exception:
            return np.asarray(_scaler.transform(X_df.values))
    return None

@st.cache_resource
def build_explainer(_model, bg, feature_names):
    if bg is None:
        return None, "Disabled (no background data)"
    try:
        return shap.LinearExplainer(_model, bg, feature_names=feature_names), "LinearExplainer"
    except Exception:
        masker = shap.maskers.Independent(bg)
        return shap.Explainer(lambda x: _model.predict_proba(x)[:, 1], masker,
                              feature_names=feature_names), "GenericExplainer"

def shap_one(explainer, x_scaled):
    try:
        exp = explainer(x_scaled)
        sv = exp.values[0] if exp.values.ndim == 2 else exp.values
        bv = float(np.ravel(exp.base_values)[0])
        return np.asarray(sv), bv
    except Exception:
        pass
    sv_raw = explainer.shap_values(x_scaled)
    sv = sv_raw[1][0] if isinstance(sv_raw, list) else sv_raw[0]
    bv = explainer.expected_value
    if isinstance(bv, (list, np.ndarray)):
        bv = bv[1] if len(bv) > 1 else bv[0]
    return np.asarray(sv), float(np.ravel(bv)[0])

bg = load_bg_data(feature_names, scaler)
explainer, explainer_type = build_explainer(model, bg, feature_names)

# 6) Header
st.title("🏥 Complicated Appendicitis Prediction System")

h1, h2, h3 = st.columns(3)
h1.metric("Model", os.path.basename(model_path))
h2.metric("Features", len(feature_names))
h3.metric("SHAP", explainer_type)

with st.expander("ℹ️ Notes", expanded=False):
    st.markdown("""
- Inputs are **standardized** using the saved scaler.
- Probability refers to **complicated appendicitis = 1**.
- SHAP computed on standardized features.
- Feature labels are displayed with **clinical units** (display only).
    """)

# Sidebar
st.sidebar.header("⚙️ Settings")
thr = st.sidebar.slider("Classification threshold", 0.05, 0.95, 0.50, 0.01)
thr_low = st.sidebar.slider("Low-risk cutoff", 0.05, 0.95, 0.30, 0.01)
thr_high = st.sidebar.slider("High-risk cutoff", 0.05, 0.95, 0.70, 0.01)
if thr_low >= thr_high:
    st.sidebar.error("Low cutoff must < High cutoff")

st.sidebar.divider()
st.sidebar.markdown("""
**📱 Install as APP:**
- **iOS**: Safari → Share → Add to Home Screen
- **Android**: Chrome → ⋮ → Install app
""")

# 7) Tabs
tab1, tab2, tab3 = st.tabs([
    "📝 Single Prediction",
    "📂 Batch Prediction",
    "📊 Model Info"
])

# ====== Tab 1: Single ======
with tab1:
    st.info("Enter patient features for risk assessment and SHAP explanation.")

    with st.form("single_form"):
        inputs = {}
        n_cols = 4 if len(feature_names) > 10 else 2
        cols = st.columns(n_cols)

        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                # ✅ 输入框显示“变量 + 单位”
                inputs[feat] = st.number_input(get_display_label(feat), value=0.0, format="%.4f")

        submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

    if submitted:
        try:
            x_df = pd.DataFrame([inputs], columns=feature_names)   # 模型仍用原始列名
            x_sc = scaler.transform(x_df)
            prob = float(model.predict_proba(x_sc)[0, 1])
            pred = int(prob >= thr)
            level, emoji, color, advice = get_risk_level(prob, thr_low, thr_high)

            st.divider()
            r1, r2, r3 = st.columns([1, 1, 2])
            r1.metric("Probability", f"{prob:.4f}")
            r2.metric("Class", "Positive" if pred else "Negative")
            r3.markdown(
                f"<div style='background:{color}15;border-left:6px solid {color};"
                f"padding:12px;border-radius:6px;'>"
                f"<b>{emoji} {level}</b><br>{advice}</div>",
                unsafe_allow_html=True
            )
            st.progress(min(prob, 1.0))

            # SHAP
            st.divider()
            st.subheader("🔍 SHAP Explanation")

            if explainer is None:
                st.warning("SHAP disabled. Place `train_data.csv` or `external_validation_data.csv` in project root.")
            else:
                with st.spinner("Computing SHAP..."):
                    sv, bv = shap_one(explainer, x_sc)

                # Waterfall（✅ 特征名显示单位）
                st.markdown("**Waterfall Plot**")
                exp_obj = shap.Explanation(
                    values=sv, base_values=bv,
                    data=x_df.iloc[0].values,
                    feature_names=display_feature_names
                )
                plt.close("all")
                fig_w = plt.figure(figsize=(11, 6.5))
                shap.plots.waterfall(exp_obj, max_display=12, show=False)
                fig_w = plt.gcf()
                st.pyplot(fig_w, use_container_width=True)
                fig_download(fig_w, "shap_waterfall.png")
                plt.close("all")

                # Force plot（✅ 特征名显示单位）
                st.markdown("**Force Plot (interactive)**")
                try:
                    fp = shap.force_plot(
                        bv, sv, x_df.iloc[0],
                        feature_names=display_feature_names,
                        matplotlib=False
                    )
                    st_shap(fp, height=200)
                except Exception as e:
                    st.info(f"Force plot unavailable: {e}")

                # Table（✅ 特征名显示单位）
                st.markdown("**Contribution Ranking**")
                ct = pd.DataFrame({
                    "Feature": display_feature_names,
                    "Raw Value": x_df.iloc[0].values,
                    "SHAP": sv,
                    "|SHAP|": np.abs(sv),
                    "Direction": np.where(sv >= 0, "↑ Risk", "↓ Protect")
                }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)
                ct.index += 1
                st.dataframe(ct, use_container_width=True, height=400)

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.text(traceback.format_exc())

    with st.expander("📏 Variable units (display labels)"):
        unit_df = pd.DataFrame({
            "Model variable": feature_names,
            "Displayed label": display_feature_names
        })
        st.dataframe(unit_df, use_container_width=True)

# ====== Tab 2: Batch ======
with tab2:
    st.info("Upload Excel (.xlsx) or CSV for batch prediction.")

    with st.expander("📥 Download template"):
        tpl = pd.DataFrame(columns=["Patient_ID"] + feature_names)
        st.download_button(
            "Download CSV template",
            tpl.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "prediction_template.csv",
            "text/csv"
        )

    uploaded = st.file_uploader("Upload file", type=["xlsx", "csv"])

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            st.success(f"✅ Loaded {len(df_up)} rows.")

            X_b = align_features(df_up, feature_names)
            X_bs = scaler.transform(X_b)
            probs = model.predict_proba(X_bs)[:, 1]
            preds = (probs >= thr).astype(int)

            df_res = df_up.copy()
            df_res["Pred_Prob"] = np.round(probs, 6)
            df_res["Pred_Class"] = preds
            df_res["Risk_Level"] = [get_risk_level(p, thr_low, thr_high)[0] for p in probs]

            st.divider()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total", len(df_res))
            s2.metric("Positive", int(preds.sum()))
            s3.metric("Negative", int((1 - preds).sum()))
            s4.metric("Mean prob", f"{probs.mean():.4f}")

            # Risk distribution chart
            plt.close("all")
            fig_bar, ax = plt.subplots(figsize=(6, 3.5))
            order = ["Low risk", "Intermediate risk", "High risk"]
            colors = ["#2ECC71", "#F39C12", "#E74C3C"]
            counts = df_res["Risk_Level"].value_counts().reindex(order, fill_value=0)
            ax.bar(order, counts.values, color=colors, edgecolor="white", lw=1.2)
            ax.set_ylabel("Count")
            ax.set_title("Risk Level Distribution")
            for i, v in enumerate(counts.values):
                ax.text(i, v + 0.3, str(int(v)), ha="center", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig_bar, use_container_width=True)
            fig_download(fig_bar, "risk_distribution.png")
            plt.close("all")

            st.dataframe(df_res, use_container_width=True, height=400)
            st.download_button(
                "💾 Download results",
                df_res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                "prediction_results.csv",
                "text/csv"
            )

            # Explain one sample
            st.divider()
            st.subheader("🔍 Explain a case")
            idx = st.selectbox(
                "Select row",
                list(df_res.index),
                format_func=lambda i: f"Row {i} | P={df_res.loc[i,'Pred_Prob']:.4f} | {df_res.loc[i,'Risk_Level']}"
            )

            if st.button("Explain", type="primary"):
                if explainer is None:
                    st.warning("SHAP disabled.")
                else:
                    x1 = X_bs[idx].reshape(1, -1)
                    with st.spinner("Computing..."):
                        sv, bv = shap_one(explainer, x1)

                    exp_obj = shap.Explanation(
                        values=sv,
                        base_values=bv,
                        data=X_b.iloc[idx].values,
                        feature_names=display_feature_names
                    )

                    plt.close("all")
                    fig = plt.figure(figsize=(11, 6.5))
                    shap.plots.waterfall(exp_obj, max_display=12, show=False)
                    fig = plt.gcf()
                    st.pyplot(fig, use_container_width=True)
                    fig_download(fig, f"shap_row_{idx}.png")
                    plt.close("all")

                    try:
                        fp = shap.force_plot(
                            bv, sv, X_b.iloc[idx],
                            feature_names=display_feature_names,
                            matplotlib=False
                        )
                        st_shap(fp, height=200)
                    except Exception:
                        pass

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.text(traceback.format_exc())

# ====== Tab 3: Model Info ======
with tab3:
    st.subheader("Model Coefficients (Logistic Regression)")

    if hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        df_coef = pd.DataFrame({
            "Feature (model)": feature_names,
            "Feature (display)": display_feature_names,
            "Coefficient": coef,
            "Odds Ratio": np.exp(coef),
            "|Coef|": np.abs(coef)
        }).sort_values("|Coef|", ascending=False).reset_index(drop=True)
        df_coef.index += 1

        st.dataframe(
            df_coef[["Feature (display)", "Coefficient", "Odds Ratio", "Feature (model)"]],
            use_container_width=True,
            height=400
        )

        plt.close("all")
        fig_c, ax = plt.subplots(figsize=(10, 7))
        dfp = df_coef.head(20).iloc[::-1]
        colors = ["#E74C3C" if v > 0 else "#0072B5" for v in dfp["Coefficient"]]
        ax.barh(dfp["Feature (display)"], dfp["Coefficient"], color=colors, alpha=0.9, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Coefficient")
        ax.set_title("Top Features (Logistic Regression)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig_c, use_container_width=True)
        fig_download(fig_c, "lr_coefficients.png")
        plt.close("all")

        st.markdown("""
**Interpretation:**
- **Positive coef** → increases risk (OR > 1)
- **Negative coef** → decreases risk (OR < 1)
        """)
    else:
        st.info("Model does not expose coefficients.")

    st.divider()
    st.markdown(f"""
| Item | Value |
|------|-------|
| Model file | `{os.path.basename(model_path)}` |
| Features | {len(feature_names)} |
| SHAP | {explainer_type} |
    """)

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#999;font-size:12px;'>"
    "⚠️ Research use only. Not for clinical diagnosis.</div>",
    unsafe_allow_html=True
)
