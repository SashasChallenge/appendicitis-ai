#python -m streamlit run .\3_streamlit.py
import streamlit as st
import streamlit.components.v1 as components  # 新增：用于显示交互式组件
import pickle
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt


# --- 0. 辅助函数：用于在Streamlit中显示交互式SHAP图 ---
def st_shap(plot, height=None):
    """
    将 SHAP 的 JS 交互图嵌入 Streamlit
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 150, scrolling=True)


# --- 1. 缓存加载资源 ---
@st.cache_resource
def load_artifacts():
    # 载入模型
    model = joblib.load("saved_models/Logistic_Regression.pkl")
    # 载入 Scaler
    try:
        scaler = joblib.load("saved_models/scaler1.pkl")
    except Exception:
        with open("saved_models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    # 载入特征名称
    try:
        feature_names = joblib.load("saved_models/feature_names1.pkl")
    except Exception:
        with open("saved_models/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

    return model, scaler, feature_names


# 初始化加载
try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("❌ 错误：找不到模型文件，请检查 saved_models/ 目录下是否有 .pkl 文件。")
    st.stop()

# --- 2. 页面配置 ---
st.set_page_config(page_title="Clinical Decision Support System", layout="wide", page_icon="🏥")
st.title("🏥 基于Logistic Regression的复杂性阑尾炎预测系统")

# --- 3. 创建标签页 (Tabs) ---
tab1, tab2 = st.tabs(["📝 单例预测 (手动输入)", "📂 批量预测 (上传Excel)"])

# ==========================================
# 模式一：单例预测 (手动输入)
# ==========================================
with tab1:
    st.info("适用于对单个样本进行快速风险评估和归因分析。")

    with st.form("single_predict_form"):
        inputs = {}
        n_cols = 4 if len(feature_names) > 10 else 2
        cols = st.columns(n_cols)

        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                inputs[feat] = st.number_input(f"{feat}", value=0.0, format="%.4f")

        submitted = st.form_submit_button("🚀 开始预测")

    if submitted:
        # 数据组装
        x_df = pd.DataFrame([inputs], columns=feature_names)

        try:
            x_scaled = scaler.transform(x_df)

            # 预测
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(x_scaled)[0, 1]
            else:
                prob = model.predict(x_scaled)[0]

            # 显示结果
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("预测结果")
                st.metric("风险概率", f"{prob * 100:.2f}%")
                if prob > 0.5:
                    st.error("🔴 高风险 (High Risk)")
                else:
                    st.success("🟢 低风险 (Low Risk)")

            # SHAP 解释
            with c2:
                st.subheader("个体归因分析")
                with st.spinner("正在计算特征贡献度..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values_all = explainer.shap_values(x_scaled)

                    if isinstance(shap_values_all, list):
                        shap_values = shap_values_all[1]
                        base_value = explainer.expected_value[1]
                    else:
                        shap_values = shap_values_all
                        base_value = explainer.expected_value
                        if isinstance(base_value, np.ndarray): base_value = base_value[0]

                    # 1. 瀑布图 (Waterfall Plot)
                    st.markdown("**1. 瀑布图 (Waterfall Plot)** - 展示累积贡献")
                    explanation = shap.Explanation(
                        values=shap_values[0],
                        base_values=base_value,
                        data=x_df.iloc[0],
                        feature_names=feature_names
                    )
                    fig = plt.figure(figsize=(10, 5))
                    shap.plots.waterfall(explanation, max_display=10, show=False)
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close(fig)

                    # 2. 力图 (Force Plot) - 新增功能
                    st.markdown("**2. 力图 (Force Plot)**")
                    st.caption("鼠标悬停在图表上可查看具体数值。")
                    # 注意：force_plot 需要 matplotlib=False 才能生成 JS 交互图
                    force_plot_html = shap.force_plot(
                        base_value,
                        shap_values[0],
                        x_df.iloc[0],
                        feature_names=feature_names,
                        matplotlib=False
                    )
                    st_shap(force_plot_html, height=160)

        except Exception as e:
            st.error(f"运行出错: {e}")
            import traceback

            st.text(traceback.format_exc())

# ==========================================
# 模式二：批量预测 (上传Excel)
# ==========================================
with tab2:
    st.info("适用于处理多条数据。请上传 Excel (.xlsx) 或 CSV 文件。")

    # 1. 下载模板
    with st.expander("📥 下载数据模板"):
        st.write("请确保您的表格包含以下列名：")
        st.code(str(feature_names), language="python")
        template_df = pd.DataFrame(columns=['Patient_ID'] + feature_names)
        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("下载 CSV 模板", csv, "prediction_template.csv", "text/csv")

    # 2. 文件上传
    uploaded_file = st.file_uploader("上传文件", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_upload = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_upload = pd.read_csv(uploaded_file, encoding='gbk')
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.write(f"✅ 成功读取 {len(df_upload)} 条数据。")

            df_upload.columns = df_upload.columns.str.strip()
            missing_cols = [col for col in feature_names if col not in df_upload.columns]

            if missing_cols:
                st.error(f"❌ 文件缺少以下必要特征列：\n{missing_cols}")
            else:
                X_batch = df_upload[feature_names]
                X_batch_scaled = scaler.transform(X_batch)

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch_scaled)[:, 1]
                else:
                    probs = model.predict(X_batch_scaled)

                df_result = df_upload.copy()
                df_result['预测概率'] = np.round(probs, 4)
                df_result['风险等级'] = ['高风险' if p > 0.5 else '低风险' for p in probs]

                st.subheader("📊 预测结果概览")
                st.dataframe(df_result.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x == '高风险' else 'background-color: #ccffcc',
                    subset=['风险等级']
                ))

                csv_result = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 下载预测结果 (.csv)", csv_result, "prediction_results.csv", "text/csv")

                # 7. 深入分析
                st.divider()
                st.subheader("🔍 深入分析：查看特定样本的SHAP解释")
                selected_index = st.selectbox(
                    "选择要分析的行号 (Index)",
                    options=df_result.index,
                    format_func=lambda x: f"行 {x} (概率: {df_result.loc[x, '预测概率']:.2%})"
                )

                if st.button("解释该样本"):
                    x_single_df = X_batch.iloc[[selected_index]]
                    x_single_scaled = X_batch_scaled[selected_index].reshape(1, -1)

                    explainer = shap.TreeExplainer(model)
                    shap_values_all = explainer.shap_values(x_single_scaled)

                    if isinstance(shap_values_all, list):
                        sv = shap_values_all[1][0]
                        bv = explainer.expected_value[1]
                    else:
                        sv = shap_values_all[0]
                        bv = explainer.expected_value
                        if isinstance(bv, np.ndarray): bv = bv[0]

                    # 1. 瀑布图
                    st.markdown("**1. 瀑布图 (Waterfall Plot)**")
                    exp = shap.Explanation(
                        values=sv, base_values=bv,
                        data=x_single_df.iloc[0], feature_names=feature_names
                    )
                    fig_batch = plt.figure(figsize=(10, 5))
                    shap.plots.waterfall(exp, max_display=10, show=False)
                    st.pyplot(fig_batch, bbox_inches='tight')
                    plt.close(fig_batch)

                    # 2. 力图 (新增)
                    st.markdown("**2. 力图 (Force Plot)**")
                    force_plot_html_batch = shap.force_plot(
                        bv,
                        sv,
                        x_single_df.iloc[0],
                        feature_names=feature_names,
                        matplotlib=False
                    )
                    st_shap(force_plot_html_batch, height=160)

        except Exception as e:
            st.error(f"处理文件时发生错误: {e}")
