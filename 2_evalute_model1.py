

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['microsoft yahei']
plt.rcParams['axes.unicode_minus'] = False



# ==========================================
# Step 1: 模型和配置加载
# ==========================================

def load_models_and_config(model_dir='saved_models'):
    """
    加载训练阶段保存的所有内容

    返回:
        models_dict: 所有训练好的模型字典（排除Voting Hard）
        scaler: 标准化器
        feature_names: 特征名称列表
    """
    print("\n" + "=" * 70)
    print("Step 1: 模型和配置加载")
    print("=" * 70)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ 模型目录不存在: {model_dir}")

    # 1. 加载标准化器
    scaler_path = os.path.join(model_dir, 'scaler1.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ 未找到标准化器: {scaler_path}")

    scaler = joblib.load(scaler_path)
    print(f"✅ 已加载标准化器: {scaler_path}")

    # 2. 加载特征名称
    feature_path = os.path.join(model_dir, 'feature_names1.pkl')
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"❌ 未找到特征名称: {feature_path}")

    feature_names = joblib.load(feature_path)
    print(f"✅ 已加载特征名称: {feature_path}")
    print(f"   特征数量: {len(feature_names)}")

    # 3. 加载所有模型（排除Voting Hard）
    models_dict = {}
    excluded_models = ['Voting_Hard', 'Voting Hard']  # 排除列表

    model_files = [f for f in os.listdir(model_dir)
                   if f.endswith('.pkl')
                   and f not in ['scaler1.pkl', 'feature_names1.pkl', 'optimized_params.pkl']
                   and not any(ex in f for ex in excluded_models)]

    if len(model_files) == 0:
        raise FileNotFoundError(f"❌ 未找到任何模型文件在: {model_dir}")

    print(f"\n📦 开始加载模型...")
    print(f"   ℹ️ 已排除: {', '.join(excluded_models)}")

    loaded_count = 0
    for model_file in sorted(model_files):
        # 将文件名转为模型名称（用空格替换下划线）
        model_name = model_file.replace('.pkl', '').replace('_', ' ')
        model_path = os.path.join(model_dir, model_file)

        try:
            model = joblib.load(model_path)

            # 检查模型是否支持概率预测
            if not hasattr(model, 'predict_proba'):
                print(f"  ⚠️ {model_name:<35} 不支持概率预测，已跳过")
                continue

            # 测试predict_proba是否可调用
            try:
                dummy_X = np.zeros((1, len(feature_names)))
                _ = model.predict_proba(dummy_X)
            except Exception:
                print(f"  ⚠️ {model_name:<35} predict_proba调用失败，已跳过")
                continue

            models_dict[model_name] = model
            loaded_count += 1
            print(f"  ✅ {model_name:<35} 加载成功")

        except Exception as e:
            print(f"  ❌ {model_name:<35} 加载失败: {str(e)}")
            continue

    if loaded_count == 0:
        raise ValueError("❌ 没有成功加载任何支持概率预测的模型")

    print(f"\n✅ 成功加载 {loaded_count} 个模型")

    return models_dict, scaler, feature_names


# ==========================================
# Step 2: 外部验证集准备
# ==========================================

def prepare_external_data(file_path, feature_names, scaler):
    """
    加载并预处理外部验证数据（假设特征完全一致）

    参数:
        file_path: 外部数据文件路径
        feature_names: 训练集的特征名称列表
        scaler: 训练集的标准化器

    返回:
        X_external: 预处理后的特征矩阵
        y_external: 标签
    """
    print("\n" + "=" * 70)
    print("Step 2: 外部验证集准备")
    print("=" * 70)

    # 1. 加载数据
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 外部数据文件不存在: {file_path}")

    print(f"\n📂 正在加载外部数据: {file_path}")

    # 尝试多种编码
    encodings = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin1']
    df = None
    used_encoding = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            used_encoding = encoding
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if df is None:
        raise ValueError(f"❌ 无法读取文件，尝试的编码: {encodings}")

    print(f"✅ 数据加载成功（编码: {used_encoding}）")
    print(f"   数据形状: {df.shape}")

    # 2. 分离特征和标签
    if 'Target' not in df.columns:
        raise ValueError("❌ 外部数据缺少 'Target' 列")

    y_external = df['Target']
    X_external_raw = df.drop(columns=['Target'])

    print(f"   正样本比例: {y_external.mean():.2%} ({y_external.sum()}/{len(y_external)})")

    # 3. 删除非数值列
    non_numeric_cols = X_external_raw.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"   删除非数值列: {non_numeric_cols}")
        X_external_raw = X_external_raw.drop(columns=non_numeric_cols)

    # 4. 特征对齐（严格按照训练集顺序）
    print(f"\n🔍 特征对齐检查...")

    # 检查是否所有训练特征都存在
    missing_features = set(feature_names) - set(X_external_raw.columns)
    if missing_features:
        raise ValueError(f"❌ 外部数据缺少以下特征: {missing_features}")

    # 按照训练集特征顺序重新排列
    X_external = X_external_raw[feature_names]

    print(f"✅ 特征对齐完成")
    print(f"   特征数量: {len(feature_names)}")
    print(f"   样本数量: {len(X_external)}")

    # 5. 标准化（只transform，不fit）
    print(f"\n🔧 应用标准化器（仅transform）...")
    X_external_scaled = scaler.transform(X_external)
    print(f"✅ 标准化完成")

    return X_external_scaled, y_external


# ==========================================
# Step 3: 模型性能评估
# ==========================================

def evaluate_models_on_external(models_dict, X_external, y_external,
                                output_dir='external_validation'):
    """
    在外部验证集上评估所有模型

    参数:
        models_dict: 所有模型的字典
        X_external: 外部验证集特征
        y_external: 外部验证集标签
        output_dir: 输出目录

    返回:
        results: 评估结果字典
        predictions: 预测结果字典
        df_results: 结果DataFrame
    """
    print("\n" + "=" * 70)
    print("Step 3: 模型性能评估（外部验证集）")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    results = {}
    predictions = {}

    print(f"\n📊 开始评估 {len(models_dict)} 个模型...")

    for idx, (model_name, model) in enumerate(models_dict.items(), 1):
        print(f"\n{'─' * 50}")
        print(f"[{idx}/{len(models_dict)}] 评估模型: {model_name}")
        print(f"{'─' * 50}")

        try:
            # 1. 预测
            y_pred = model.predict(X_external)
            y_prob = model.predict_proba(X_external)[:, 1]

            # 2. 基础指标
            accuracy = accuracy_score(y_external, y_pred)
            precision = precision_score(y_external, y_pred, zero_division=0)
            recall = recall_score(y_external, y_pred, zero_division=0)
            f1 = f1_score(y_external, y_pred, zero_division=0)

            # 3. 混淆矩阵
            cm = confusion_matrix(y_external, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # 4. 派生指标
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            mcc = matthews_corrcoef(y_external, y_pred)
            kappa = cohen_kappa_score(y_external, y_pred)
            youden = sensitivity + specificity - 1

            # 5. 概率指标
            auc = roc_auc_score(y_external, y_prob)
            brier = brier_score_loss(y_external, y_prob)
            ap = average_precision_score(y_external, y_prob)

            # 6. 保存结果
            results[model_name] = {
                'AUC': auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'PPV': ppv,
                'NPV': npv,
                'Brier Score': brier,
                'MCC': mcc,
                'Kappa': kappa,
                'AP': ap,
                'Youden Index': youden,
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn)
            }

            predictions[model_name] = {
                'y_pred': y_pred,
                'y_prob': y_prob
            }

            # 7. 打印关键指标
            print(f"  AUC: {auc:.4f} | Acc: {accuracy:.4f} | F1: {f1:.4f}")
            print(f"  Sen: {sensitivity:.4f} | Spe: {specificity:.4f}")

        except Exception as e:
            print(f"  ❌ 评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n" + "=" * 70)
    print(f"✅ 评估完成！成功评估 {len(results)}/{len(models_dict)} 个模型")
    print(f"=" * 70)

    # 8. 生成结果表
    if len(results) == 0:
        raise ValueError("❌ 没有成功评估任何模型")

    print(f"\n📊 生成结果汇总...")
    df_results = pd.DataFrame(results).T

    # 排序列
    key_metrics = ['AUC', 'Accuracy', 'F1-Score', 'Sensitivity', 'Specificity',
                   'Precision', 'Recall', 'PPV', 'NPV', 'MCC', 'Kappa', 'Brier Score', 'AP',
                   'Youden Index', 'TP', 'TN', 'FP', 'FN']

    cols = [col for col in key_metrics if col in df_results.columns]
    df_results = df_results[cols].round(4).sort_values('AUC', ascending=False)

    # 保存CSV
    results_path = os.path.join(output_dir, 'external_results.csv')
    df_results.to_csv(results_path, encoding='utf-8-sig')
    print(f"✅ 结果已保存: {results_path}")

    # 打印Top 5
    print(f"\n🏆 Top 5 模型（按AUC排序）:")
    top5_cols = ['AUC', 'Accuracy', 'F1-Score', 'Sensitivity', 'Specificity']
    print(df_results[top5_cols].head(5).to_string())

    # 打印统计摘要（修正语法错误）
    print(f"\n📈 性能统计摘要:")
    summary_metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']
    # 修正：使用 .loc 而不是双点
    summary_stats = df_results[summary_metrics].describe().loc[['mean', 'std', 'min', 'max']]
    print(summary_stats.round(4).to_string())

    return results, predictions, df_results


# ==========================================
# 执行前3步流程（直接运行）
# ==========================================

print("\n" + "=" * 70)
print("🚀 外部验证集评估流程 - 前3步")
print("=" * 70)

# Step 1: 加载模型和配置
models_dict, scaler, feature_names = load_models_and_config(
    model_dir='saved_models'
)

# Step 2: 准备外部验证数据
external_file = 'external_validation_data.csv'
X_external, y_external = prepare_external_data(
    file_path=external_file,
    feature_names=feature_names,
    scaler=scaler
)

# Step 3: 模型性能评估
results, predictions, df_results = evaluate_models_on_external(
    models_dict=models_dict,
    X_external=X_external,
    y_external=y_external,
    output_dir='external_validation'
)

print("\n" + "=" * 70)
print("✅ 前3步完成！")
print("=" * 70)
print("\n生成的文件:")
print("  📄 external_validation/external_results.csv")
print("\n📌 数据说明:")
print(f"  - 训练集特征: {len(feature_names)}")
print(f"  - 外部验证集样本: {len(X_external)}")
print(f"  - 评估模型数: {len(results)}")
print(f"  - 最优模型: {df_results.index[0]}")
print(f"  - 最优AUC: {df_results['AUC'].iloc[0]:.4f}")


# ==========================================
# Step 4: 训练 vs 外部验证对比
# ==========================================

def compare_train_external_results(train_results_path, external_results_path,
                                   output_dir='external_validation'):
    """
    对比训练集和外部验证集的结果

    参数:
        train_results_path: 训练结果CSV路径
        external_results_path: 外部验证结果CSV路径
        output_dir: 输出目录

    返回:
        comparison_df: 对比结果DataFrame
    """
    print("\n" + "=" * 70)
    print("Step 4: 训练 vs 外部验证对比")
    print("=" * 70)

    # 1. 加载训练集结果
    if not os.path.exists(train_results_path):
        raise FileNotFoundError(f"❌ 训练结果文件不存在: {train_results_path}")

    df_train = pd.read_csv(train_results_path, index_col=0)
    print(f"✅ 已加载训练集结果: {train_results_path}")
    print(f"   训练集模型数: {len(df_train)}")

    # 2. 加载外部验证集结果
    if not os.path.exists(external_results_path):
        raise FileNotFoundError(f"❌ 外部验证结果文件不存在: {external_results_path}")

    df_external = pd.read_csv(external_results_path, index_col=0)
    print(f"✅ 已加载外部验证结果: {external_results_path}")
    print(f"   外部验证模型数: {len(df_external)}")

    # 3. 找到共同的模型
    common_models = list(set(df_train.index) & set(df_external.index))

    if len(common_models) == 0:
        raise ValueError("❌ 训练集和外部验证集没有共同的模型")

    print(f"\n🔍 共同模型数: {len(common_models)}")

    # 4. 计算对比指标
    print(f"\n📊 计算对比指标...")

    comparison_data = []

    for model_name in common_models:
        train_row = df_train.loc[model_name]
        external_row = df_external.loc[model_name]

        # 关键指标对比
        train_auc = train_row['AUC']
        external_auc = external_row['AUC']
        auc_diff = train_auc - external_auc

        # 过拟合程度
        overfitting_degree = auc_diff / train_auc if train_auc > 0 else 0

        # 泛化能力评分 (越接近1越好)
        generalization_score = 1 - abs(overfitting_degree)

        # 其他指标差异
        acc_diff = train_row['Accuracy'] - external_row['Accuracy']
        f1_diff = train_row['F1-Score'] - external_row['F1-Score']
        sen_diff = train_row['Sensitivity'] - external_row['Sensitivity']
        spe_diff = train_row['Specificity'] - external_row['Specificity']

        # 综合稳定性评分
        stability_score = 1 - np.mean([
            abs(auc_diff / train_auc) if train_auc > 0 else 0,
            abs(acc_diff / train_row['Accuracy']) if train_row['Accuracy'] > 0 else 0,
            abs(f1_diff / train_row['F1-Score']) if train_row['F1-Score'] > 0 else 0
        ])

        comparison_data.append({
            'Model': model_name,
            'Train_AUC': train_auc,
            'External_AUC': external_auc,
            'AUC_Diff': auc_diff,
            'Overfitting_Degree': overfitting_degree,
            'Generalization_Score': generalization_score,
            'Train_Accuracy': train_row['Accuracy'],
            'External_Accuracy': external_row['Accuracy'],
            'Accuracy_Diff': acc_diff,
            'Train_F1': train_row['F1-Score'],
            'External_F1': external_row['F1-Score'],
            'F1_Diff': f1_diff,
            'Train_Sensitivity': train_row['Sensitivity'],
            'External_Sensitivity': external_row['Sensitivity'],
            'Sensitivity_Diff': sen_diff,
            'Train_Specificity': train_row['Specificity'],
            'External_Specificity': external_row['Specificity'],
            'Specificity_Diff': spe_diff,
            'Stability_Score': stability_score
        })

    # 5. 创建对比DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Generalization_Score', ascending=False)

    # 6. 保存对比结果
    comparison_path = os.path.join(output_dir, 'comparison_results.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"✅ 对比结果已保存: {comparison_path}")

    # 7. 打印关键发现
    print(f"\n📈 关键发现:")

    # 最佳泛化能力
    best_gen_model = comparison_df.iloc[0]
    print(f"\n🏆 最佳泛化能力: {best_gen_model['Model']}")
    print(f"   训练AUC: {best_gen_model['Train_AUC']:.4f}")
    print(f"   外部AUC: {best_gen_model['External_AUC']:.4f}")
    print(f"   AUC下降: {best_gen_model['AUC_Diff']:.4f}")
    print(f"   泛化评分: {best_gen_model['Generalization_Score']:.4f}")

    # 最严重过拟合
    worst_gen_model = comparison_df.iloc[-1]
    print(f"\n⚠️ 过拟合最严重: {worst_gen_model['Model']}")
    print(f"   训练AUC: {worst_gen_model['Train_AUC']:.4f}")
    print(f"   外部AUC: {worst_gen_model['External_AUC']:.4f}")
    print(f"   AUC下降: {worst_gen_model['AUC_Diff']:.4f}")
    print(f"   过拟合程度: {worst_gen_model['Overfitting_Degree']:.2%}")

    # 统计摘要
    print(f"\n📊 整体统计:")
    print(f"   平均AUC下降: {comparison_df['AUC_Diff'].mean():.4f}")
    print(f"   AUC下降标准差: {comparison_df['AUC_Diff'].std():.4f}")
    print(f"   平均泛化评分: {comparison_df['Generalization_Score'].mean():.4f}")
    print(f"   显著过拟合模型数 (AUC下降>0.10): {(comparison_df['AUC_Diff'] > 0.10).sum()}")

    return comparison_df


# ==========================================
# Step 5: 基础对比可视化
# ==========================================
# ==========================================
# Step 5-6: 终极优化版（SCI期刊级 + 多格式 + 正确口径）
# ==========================================
# ==========================================
# Step 5-6: 最终版（SCI期刊级 + 多格式 + 全面美化）
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# ========== Lancet/NEJM 配色 ==========
SCI = {
    "colors": ["#0072B5", "#BC3C29", "#20854E", "#E18727",
               "#6F99AD", "#925E9F", "#FFDC91", "#7876B1",
               "#EE4C97", "#008B45"],
    "bg": "#FAFAFA",
    "grid": "#D5D5D5",
    "text": "#333333",
    "card": "#F7F7F7",
    "border": "#CCCCCC",
}

def sci_style():
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "microsoft yahei",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#555555",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 9,
        "legend.framealpha": 0.92,
        "legend.edgecolor": SCI["border"],
        "grid.alpha": 0.3,
        "grid.color": SCI["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.2,
        "lines.antialiased": True,
    })

def sci_despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

def sci_save(fig, out_dir, name, fmts=("png", "pdf", "svg"), dpi=600):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for f in fmts:
        p = os.path.join(out_dir, f"{name}.{f}")
        fig.savefig(p, bbox_inches="tight",
                    dpi=(dpi if f in ("png", "tif", "tiff") else None),
                    facecolor="white", edgecolor="none")
        saved.append(f)
    plt.close(fig)
    print(f"   ✅ {name} -> {', '.join(saved)}")

def sci_palette(n):
    base = SCI["colors"]
    return base[:n] if n <= len(base) else sns.color_palette("husl", n)

def sci_watermark(ax, text="", alpha=0.03):
    """右下角淡水印（可选）"""
    if text:
        ax.text(0.98, 0.02, text, transform=ax.transAxes,
                fontsize=7, alpha=alpha, ha="right", va="bottom", color="gray")

# ========== 重建内部测试集 ==========
def rebuild_internal_test(train_csv, feature_names, scaler,
                          test_size=0.2, random_state=42):
    encs = ['utf-8', 'gbk', 'gb18030', 'latin1']
    df = None
    for e in encs:
        try:
            df = pd.read_csv(train_csv, encoding=e)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"无法读取 {train_csv}")

    y = df['Target']
    X = df.drop(columns=['Target'])
    non_num = X.select_dtypes(include=['object']).columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_test = X_test[feature_names]
    X_test_scaled = scaler.transform(X_test)
    print(f"✅ 内部测试集重建完成: {X_test_scaled.shape[0]} 样本")
    return X_test_scaled, y_test

def get_preds(models_dict, X, y):
    preds = {}
    for name, model in models_dict.items():
        if not hasattr(model, "predict_proba"):
            continue
        try:
            yp = model.predict_proba(X)[:, 1]
            preds[name] = {"y_prob": yp, "y_pred": model.predict(X)}
        except Exception:
            continue
    return preds

# ==========================================
# Step 5: 基础对比可视化
# ==========================================
# ==========================================
# Step 5-6: 最终版（大字体 + SCI期刊级 + 多格式）
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# ========== 配色 ==========
SCI = {
    "colors": ["#0072B5", "#BC3C29", "#20854E", "#E18727",
               "#6F99AD", "#925E9F", "#FFDC91", "#7876B1",
               "#EE4C97", "#008B45"],
    "card": "#F7F7F7",
    "border": "#CCCCCC",
    "grid": "#D5D5D5",
    "text": "#333333",
}

def sci_style():
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "microsoft yahei",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        # ===== 全局字号（加大） =====
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 13,
        "legend.framealpha": 0.92,
        "legend.edgecolor": SCI["border"],
        "figure.titlesize": 20,
        # ===== 线条 =====
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#555555",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.alpha": 0.3,
        "grid.color": SCI["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.2,
        "lines.antialiased": True,
    })

def sci_despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

def sci_save(fig, out_dir, name, fmts=("png", "pdf", "svg"), dpi=600):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for f in fmts:
        p = os.path.join(out_dir, f"{name}.{f}")
        fig.savefig(p, bbox_inches="tight",
                    dpi=(dpi if f in ("png", "tif", "tiff") else None),
                    facecolor="white", edgecolor="none")
        saved.append(f)
    plt.close(fig)
    print(f"   ✅ {name} -> {', '.join(saved)}")

def sci_palette(n):
    base = SCI["colors"]
    return base[:n] if n <= len(base) else sns.color_palette("husl", n)

# ========== 重建内部测试集 ==========
def rebuild_internal_test(train_csv, feature_names, scaler,
                          test_size=0.2, random_state=42):
    encs = ['utf-8', 'gbk', 'gb18030', 'latin1']
    df = None
    for e in encs:
        try:
            df = pd.read_csv(train_csv, encoding=e)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"无法读取 {train_csv}")

    y = df['Target']
    X = df.drop(columns=['Target'])
    non_num = X.select_dtypes(include=['object']).columns.tolist()
    if non_num:
        X = X.drop(columns=non_num)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_test = X_test[feature_names]
    X_test_scaled = scaler.transform(X_test)
    print(f"✅ 内部测试集重建完成: {X_test_scaled.shape[0]} 样本")
    return X_test_scaled, y_test

def get_preds(models_dict, X, y):
    preds = {}
    for name, model in models_dict.items():
        if not hasattr(model, "predict_proba"):
            continue
        try:
            yp = model.predict_proba(X)[:, 1]
            preds[name] = {"y_prob": yp, "y_pred": model.predict(X)}
        except Exception:
            continue
    return preds

# ==========================================
# Step 5: 基础对比可视化
# ==========================================
def plot_basic_comparison_figures(
    df_train, df_external, comparison_df, trained_models,
    preds_internal, preds_external,
    y_internal, y_external,
    output_dir='external_validation',
    fmts=("png", "pdf", "svg")
):
    print("\n" + "=" * 70)
    print("Step 5: 基础对比可视化（大字体最终版）")
    print("=" * 70)

    d = os.path.join(output_dir, 'basic_comparison')
    os.makedirs(d, exist_ok=True)

    print("\n📊 1/5 ROC 对比...")
    _plot_01_roc(df_train, df_external, preds_internal, preds_external,
                 y_internal, y_external, d, fmts)
    print("📊 2/5 指标对比柱状图...")
    _plot_02_bar(df_train, df_external, comparison_df, d, fmts)
    print("📊 3/5 泛化能力散点图...")
    _plot_03_scatter(comparison_df, d, fmts)
    print("📊 4/5 校准曲线...")
    _plot_04_calib(df_external, preds_external, y_external, d, fmts)
    print("📊 5/5 混淆矩阵对比...")
    _plot_05_cm(df_train, df_external, preds_internal, preds_external,
                y_internal, y_external, d, fmts)
    print(f"\n✅ 基础对比完成！→ {d}/")

# ---------- 01 ROC ----------
def _plot_01_roc(df_tr, df_ex, pi, pe, yi, ye, d, fmts):
    sci_style()
    top5_tr = df_tr.sort_values("AUC", ascending=False).head(5).index.tolist()
    top5_ex = df_ex.sort_values("AUC", ascending=False).head(5).index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.5))

    # 左: Internal
    for name, c in zip(top5_tr, sci_palette(5)):
        if name in pi:
            fpr, tpr, _ = roc_curve(yi, pi[name]["y_prob"])
            auc_val = df_tr.loc[name, "AUC"]
            ax1.plot(fpr, tpr, color=c, lw=2.8, label=f"{name} ({auc_val:.3f})")
            ax1.fill_between(fpr, tpr, alpha=0.04, color=c)

    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.4)
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlabel("1 − Specificity (FPR)", fontsize=15)
    ax1.set_ylabel("Sensitivity (TPR)", fontsize=15)
    ax1.set_title("Internal Test ROC\n(Top 5 by Internal AUC)", fontsize=16, pad=14)
    ax1.legend(loc="lower right", fontsize=11, frameon=True,
               title="Model (AUC)", title_fontsize=12)
    ax1.tick_params(labelsize=13)
    ax1.grid(True, alpha=0.2); sci_despine(ax1)

    # 右: External
    for name, c in zip(top5_ex, sci_palette(5)):
        if name in pe:
            fpr, tpr, _ = roc_curve(ye, pe[name]["y_prob"])
            auc_val = df_ex.loc[name, "AUC"]
            ax2.plot(fpr, tpr, color=c, lw=2.8, label=f"{name} ({auc_val:.3f})")
            ax2.fill_between(fpr, tpr, alpha=0.04, color=c)

    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.4)
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("1 − Specificity (FPR)", fontsize=15)
    ax2.set_ylabel("Sensitivity (TPR)", fontsize=15)
    ax2.set_title("External Validation ROC\n(Top 5 by External AUC)", fontsize=16, pad=14)
    ax2.legend(loc="lower right", fontsize=11, frameon=True,
               title="Model (AUC)", title_fontsize=12)
    ax2.tick_params(labelsize=13)
    ax2.grid(True, alpha=0.2); sci_despine(ax2)

    fig.suptitle("ROC Curves: Internal Test vs External Validation",
                 fontsize=19, fontweight="bold", y=1.02)
    plt.tight_layout()
    sci_save(fig, d, "01_comparison_roc_curves", fmts)

# ---------- 02 Metrics bar ----------
def _plot_02_bar(df_tr, df_ex, comp_df, d, fmts):
    sci_style()
    top10 = comp_df.sort_values("Generalization_Score", ascending=False).head(10)["Model"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    metrics = ["AUC", "Accuracy", "Sensitivity", "Specificity"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        tv = [df_tr.loc[m, metric] for m in top10]
        ev = [df_ex.loc[m, metric] for m in top10]

        x = np.arange(len(top10))
        w = 0.38

        b1 = ax.barh(x - w / 2, tv, w, label="Internal Test",
                      color=SCI["colors"][0], alpha=0.88, edgecolor="white", lw=0.6)
        b2 = ax.barh(x + w / 2, ev, w, label="External",
                      color=SCI["colors"][1], alpha=0.88, edgecolor="white", lw=0.6)

        ax.set_yticks(x)
        ax.set_yticklabels([m[:26] for m in top10], fontsize=12)
        ax.set_xlabel(metric, fontsize=14)
        ax.set_title(f"{metric} (Top 10 by Generalization Score)", fontsize=15, pad=10)
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.15, axis="x"); sci_despine(ax)

        for bars in [b1, b2]:
            for bar in bars:
                v = bar.get_width()
                ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=10, color=SCI["text"])

    fig.suptitle("Performance Metrics: Internal Test vs External Validation",
                 fontsize=18, fontweight="bold", y=1.0)
    plt.tight_layout()
    sci_save(fig, d, "02_comparison_metrics_bar", fmts)

# ---------- 03 Scatter ----------
def _plot_03_scatter(comp_df, d, fmts):
    sci_style()
    fig, ax = plt.subplots(figsize=(11, 10))

    sc = ax.scatter(
        comp_df["Train_AUC"], comp_df["External_AUC"],
        c=comp_df["Generalization_Score"], s=220, alpha=0.85,
        edgecolors="black", lw=0.7, cmap="RdYlGn", vmin=0.8, vmax=1.0, zorder=3
    )

    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1.5, alpha=0.5, label="Perfect generalization")
    ax.plot([0.5, 1], [0.4, 0.9], color=SCI["colors"][1], ls=":", lw=1.2, alpha=0.5,
            label="Overfitting warning (Δ=0.10)")

    xx = np.linspace(0.5, 1, 100)
    ax.fill_between(xx, xx - 0.05, xx + 0.05, alpha=0.06, color=SCI["colors"][2],
                    label="±0.05 zone")

    for _, row in comp_df.iterrows():
        ax.annotate(
            row["Model"][:18],
            (row["Train_AUC"], row["External_AUC"]),
            fontsize=10, alpha=0.85, xytext=(5, 5), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5)
        )

    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Generalization Score", fontsize=13)
    cb.ax.tick_params(labelsize=12)

    ax.set_xlabel("Internal Test AUC", fontsize=15)
    ax.set_ylabel("External Validation AUC", fontsize=15)
    ax.set_title("Generalization Ability Analysis\n"
                 "(Closer to diagonal = better generalization)", fontsize=16, pad=14)
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(0.5, 1.02); ax.set_ylim(0.5, 1.02)
    ax.set_aspect("equal")
    sci_despine(ax)
    plt.tight_layout()
    sci_save(fig, d, "03_train_vs_external_scatter", fmts)

# ---------- 04 Calibration ----------
def _plot_04_calib(df_ex, pe, ye, d, fmts):
    sci_style()
    top4 = df_ex.sort_values("AUC", ascending=False).head(4).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    colors = sci_palette(4)

    for i, (name, c) in enumerate(zip(top4, colors)):
        ax = axes[i]
        yp = pe[name]["y_prob"]
        fp, mp = calibration_curve(ye, yp, n_bins=10, strategy="uniform")
        brier = brier_score_loss(ye, yp)

        ax.plot(mp, fp, "o-", color=c, lw=3, markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3, label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Perfect")

        # 置信带
        for xp, yp_pt in zip(mp, fp):
            n_bin = max(1, int(len(yp) / 10))
            se = np.sqrt(yp_pt * (1 - yp_pt) / n_bin)
            ax.fill_between([xp - 0.02, xp + 0.02],
                            max(0, yp_pt - 1.96 * se), min(1, yp_pt + 1.96 * se),
                            alpha=0.12, color=c)

        # 概率分布
        ax2 = ax.twinx()
        ax2.hist(yp[np.asarray(ye) == 0], bins=25, range=(0, 1),
                 alpha=0.18, color=SCI["colors"][0])
        ax2.hist(yp[np.asarray(ye) == 1], bins=25, range=(0, 1),
                 alpha=0.18, color=SCI["colors"][1])
        ax2.set_ylim(0, ax2.get_ylim()[1] * 5)
        ax2.set_yticks([]); ax2.spines["right"].set_visible(False)

        ax.text(0.04, 0.96,
                f"AUC = {df_ex.loc[name, 'AUC']:.3f}\nBrier = {brier:.4f}",
                transform=ax.transAxes, va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.92, ec=SCI["border"]))

        ax.set_title(f"#{i + 1}  {name}", fontsize=15, pad=12)
        ax.set_xlabel("Predicted Probability", fontsize=14)
        ax.set_ylabel("Observed Proportion", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.15); sci_despine(ax)
        if i == 0:
            ax.legend(loc="lower right", fontsize=11)

    fig.suptitle("Calibration Curves — External Validation (Top 4 by External AUC)",
                 fontsize=17, fontweight="bold", y=0.998)
    plt.tight_layout()
    sci_save(fig, d, "04_comparison_calibration", fmts)

# ---------- 05 CM ----------
def _plot_05_cm(df_tr, df_ex, pi, pe, yi, ye, d, fmts):
    sci_style()
    best_tr = df_tr["AUC"].idxmax()
    best_ex = df_ex["AUC"].idxmax()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    if best_tr in pi:
        cm1 = confusion_matrix(yi, pi[best_tr]["y_pred"])
    else:
        r = df_tr.loc[best_tr]
        cm1 = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]]).astype(int)

    cm2 = confusion_matrix(ye, pe[best_ex]["y_pred"])

    _draw_cm(ax1, cm1,
             f"Internal Test — Best Model\n{best_tr}\nAUC = {df_tr.loc[best_tr, 'AUC']:.3f}",
             "Blues")
    _draw_cm(ax2, cm2,
             f"External Validation — Best Model\n{best_ex}\nAUC = {df_ex.loc[best_ex, 'AUC']:.3f}",
             "Oranges")

    fig.suptitle("Confusion Matrix — Best Model Comparison",
                 fontsize=19, fontweight="bold", y=1.03)
    plt.tight_layout()
    sci_save(fig, d, "05_comparison_confusion_matrix", fmts)

def _draw_cm(ax, cm, title, cmap):
    total = cm.sum()
    pct = cm / total * 100
    annot = np.array([
        [f"{cm[0, 0]}\n({pct[0, 0]:.1f}%)", f"{cm[0, 1]}\n({pct[0, 1]:.1f}%)"],
        [f"{cm[1, 0]}\n({pct[1, 0]:.1f}%)", f"{cm[1, 1]}\n({pct[1, 1]:.1f}%)"]
    ])
    sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, ax=ax,
                xticklabels=["Pred Neg", "Pred Pos"],
                yticklabels=["Act Neg", "Act Pos"],
                linewidths=2, linecolor="white", cbar=False,
                annot_kws={"size": 17, "weight": "bold"})
    ax.set_title(title, fontsize=14, pad=14)
    ax.tick_params(labelsize=13)

    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn) if (tp + fn) else 0
    spe = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    acc = (tp + tn) / total

    card = (f"Accuracy  {acc:.3f}\n"
            f"Sensitivity {sen:.3f}\n"
            f"Specificity {spe:.3f}\n"
            f"PPV       {ppv:.3f}\n"
            f"NPV       {npv:.3f}\n"
            f"─────────\n"
            f"TP={tp}  FP={fp}\n"
            f"FN={fn}  TN={tn}")
    ax.text(1.06, 0.5, card, transform=ax.transAxes, fontsize=12, va="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=SCI["card"], ec=SCI["border"], lw=0.7))



# ==========================================
# Step 6: 医学期刊级别对比可视化
# ==========================================
# ==========================================
# Step 6: 医学期刊级别对比可视化
# ==========================================
def plot_medical_comparison_figures(
    df_train, df_external, comparison_df,
    preds_internal, preds_external,
    y_internal, y_external,
    output_dir='external_validation',
    fmts=("png", "pdf", "svg")
):
    print("\n" + "=" * 70)
    print("Step 6: 医学期刊级别对比可视化（大字体最终版）")
    print("=" * 70)

    d = os.path.join(output_dir, 'medical_comparison')
    os.makedirs(d, exist_ok=True)

    print("\n📊 1/5 DCA 对比...")
    _plot_m01_dca(df_train, df_external, preds_internal, preds_external,
                  y_internal, y_external, d, fmts)
    print("📊 2/5 校准 + H-L...")
    _plot_m02_hl(df_external, preds_external, y_external, d, fmts)
    print("📊 3/5 Forest Plot...")
    _plot_m03_forest(df_train, df_external, d, fmts)
    print("📊 4/5 瀑布图...")
    _plot_m04_waterfall(comparison_df, d, fmts)
    print("📊 5/5 雷达图...")
    _plot_m05_radar(df_train, df_external, comparison_df, d, fmts)
    print(f"\n✅ 医学期刊可视化完成！→ {d}/")

# ---------- M01 DCA ----------
def _calc_nb(y_true, y_prob, thr):
    yp = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
    n = len(y_true)
    if thr >= 1.0:
        return 0.0
    return (tp / n) - (fp / n) * (thr / (1 - thr))

def _plot_m01_dca(df_tr, df_ex, pi, pe, yi, ye, d, fmts):
    sci_style()
    top5_tr = df_tr.sort_values("AUC", ascending=False).head(5).index.tolist()
    top5_ex = df_ex.sort_values("AUC", ascending=False).head(5).index.tolist()
    thr = np.arange(0.01, 0.99, 0.01)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8.5))

    for ax, top_list, preds, y_true, df_ref, side in [
        (ax1, top5_tr, pi, yi, df_tr, "Internal Test DCA\n(Top 5 by Internal AUC)"),
        (ax2, top5_ex, pe, ye, df_ex, "External Validation DCA\n(Top 5 by External AUC)")
    ]:
        colors = sci_palette(len(top_list))
        for name, c in zip(top_list, colors):
            if name in preds:
                nb = [_calc_nb(y_true, preds[name]["y_prob"], t) for t in thr]
                ax.plot(thr, nb, color=c, lw=2.8,
                        label=f"{name} ({df_ref.loc[name, 'AUC']:.3f})")

        prev = float(np.mean(y_true))
        ta = [prev - (1 - prev) * (t / (1 - t)) for t in thr]
        ax.plot(thr, ta, "k--", lw=1.8, label="Treat All")
        ax.axhline(0, color="gray", ls=":", lw=1.8, label="Treat None")
        ax.axvspan(0.1, 0.4, alpha=0.04, color=SCI["colors"][0])

        ax.set_xlim(0, 0.8)
        ax.set_ylim(-0.05, 0.30)
        ax.set_yticks(np.arange(-0.05, 0.301, 0.05))
        ax.set_xlabel("Threshold Probability", fontsize=15)
        ax.set_ylabel("Net Benefit", fontsize=15)
        ax.set_title(side, fontsize=15, pad=12)
        ax.legend(loc="upper right", fontsize=10, frameon=True,
                  title="Model (AUC)", title_fontsize=11)
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.2)
        sci_despine(ax)

    fig.suptitle("Decision Curve Analysis: Internal Test vs External Validation",
                 fontsize=19, fontweight="bold", y=1.02)
    plt.tight_layout()
    sci_save(fig, d, "01_comparison_dca", fmts)

# ---------- M02 Calibration + H-L ----------
def _hl_test(yt, yp, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    bi = np.clip(np.digitize(yp, bins[:-1]) - 1, 0, nb - 1)
    yt, yp = np.asarray(yt), np.asarray(yp)
    obs, exp, cnt = np.zeros(nb), np.zeros(nb), np.zeros(nb)
    for i in range(nb):
        m = bi == i; cnt[i] = m.sum()
        if cnt[i] > 0:
            obs[i] = (yt[m] == 1).sum(); exp[i] = yp[m].sum()
    m = cnt > 0
    chi2 = np.sum((obs[m] - exp[m]) ** 2 / (exp[m] * (1 - exp[m] / cnt[m]) + 1e-10))
    return float(chi2), float(1 - stats.chi2.cdf(chi2, nb - 2))

def _plot_m02_hl(df_ex, pe, ye, d, fmts):
    sci_style()
    top6 = df_ex.sort_values("AUC", ascending=False).head(6).index.tolist()

    nc, nr = 3, int(np.ceil(len(top6) / 3))
    fig, axes = plt.subplots(nr, nc, figsize=(21, 7.5 * nr))
    axes = np.array(axes).reshape(-1)
    colors = sci_palette(len(top6))

    for i, (name, c) in enumerate(zip(top6, colors)):
        ax = axes[i]
        yp = pe[name]["y_prob"]
        fp, mp = calibration_curve(ye, yp, n_bins=10, strategy="uniform")
        chi2, p = _hl_test(ye, yp)
        brier = brier_score_loss(ye, yp)

        ax.plot(mp, fp, "o-", color=c, lw=3, markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5)

        # 置信带
        for xp, yp_pt in zip(mp, fp):
            n_bin = max(1, int(len(yp) / 10))
            se = np.sqrt(yp_pt * (1 - yp_pt) / n_bin)
            ax.fill_between([xp - 0.02, xp + 0.02],
                            max(0, yp_pt - 1.96 * se), min(1, yp_pt + 1.96 * se),
                            alpha=0.12, color=c)

        # 概率分布
        ax2 = ax.twinx()
        ax2.hist(yp[np.asarray(ye) == 0], bins=25, range=(0, 1),
                 alpha=0.15, color=SCI["colors"][0])
        ax2.hist(yp[np.asarray(ye) == 1], bins=25, range=(0, 1),
                 alpha=0.15, color=SCI["colors"][1])
        ax2.set_ylim(0, ax2.get_ylim()[1] * 5)
        ax2.set_yticks([]); ax2.spines["right"].set_visible(False)

        hl_tag = "Good" if p > 0.05 else "Poor"
        ax.text(0.04, 0.96,
                f"AUC = {df_ex.loc[name, 'AUC']:.3f}\n"
                f"H-L χ² = {chi2:.2f}\n"
                f"p = {p:.4f} ({hl_tag})\n"
                f"Brier = {brier:.4f}",
                transform=ax.transAxes, va="top", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.92, ec=SCI["border"]))

        ax.set_title(f"#{i + 1}  {name}", fontsize=14, pad=12)
        ax.set_xlabel("Predicted Probability", fontsize=14)
        ax.set_ylabel("Observed Proportion", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.15); sci_despine(ax)

    for j in range(len(top6), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Calibration + Hosmer-Lemeshow — External (Top 6 by External AUC)",
                 fontsize=17, fontweight="bold", y=0.998)
    plt.tight_layout()
    sci_save(fig, d, "02_comparison_calibration_hl", fmts)

# ---------- M03 Forest ----------
def _plot_m03_forest(df_tr, df_ex, d, fmts):
    sci_style()
    common = sorted(set(df_tr.index) & set(df_ex.index),
                    key=lambda x: df_ex.loc[x, "AUC"], reverse=True)[:10]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 11))

    for ax, df, title, color in [
        (ax1, df_tr, "Internal Test", SCI["colors"][0]),
        (ax2, df_ex, "External Validation", SCI["colors"][2])
    ]:
        dfp = df.loc[common].sort_values("AUC", ascending=True)
        y = np.arange(len(dfp))
        aucs = dfp["AUC"].values

        if "CV_AUC_std" in dfp.columns:
            lo = np.clip(aucs - 1.96 * dfp["CV_AUC_std"].values, 0, 1)
            hi = np.clip(aucs + 1.96 * dfp["CV_AUC_std"].values, 0, 1)
        else:
            lo = np.clip(aucs - 0.05, 0, 1)
            hi = np.clip(aucs + 0.05, 0, 1)

        for i in range(len(aucs)):
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color="#888888", lw=2, zorder=1)
            ax.plot([lo[i], lo[i]], [y[i] - 0.12, y[i] + 0.12], color="#888888", lw=1.2)
            ax.plot([hi[i], hi[i]], [y[i] - 0.12, y[i] + 0.12], color="#888888", lw=1.2)

        ax.scatter(aucs, y, s=120, color=color, edgecolors="black",
                   lw=0.7, zorder=3, marker="D")

        ax.axvline(0.5, color="red", ls="--", lw=1, alpha=0.4)
        ax.axvline(0.8, color=SCI["colors"][2], ls=":", lw=0.8, alpha=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(dfp.index, fontsize=12)
        ax.set_xlabel("AUC (95% CI)", fontsize=14)
        ax.set_title(title, fontsize=15, pad=12)
        ax.set_xlim(0.45, 1.05)
        ax.tick_params(labelsize=12)
        ax.grid(True, axis="x", alpha=0.15)
        sci_despine(ax)

        for i, v in enumerate(aucs):
            ci_str = f"{lo[i]:.3f}–{hi[i]:.3f}"
            ax.text(1.06, y[i], f"{v:.3f} ({ci_str})",
                    va="center", fontsize=10, fontfamily="monospace",
                    transform=ax.get_yaxis_transform())

    fig.suptitle("Forest Plot: Internal Test vs External Validation",
                 fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    sci_save(fig, d, "03_comparison_forest_plot", fmts)

# ---------- M04 Waterfall ----------
def _plot_m04_waterfall(comp_df, d, fmts):
    sci_style()
    dfp = comp_df.sort_values("AUC_Diff", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(15, max(9, 0.5 * len(dfp) + 2)))

    colors = [SCI["colors"][1] if x > 0.05
              else "#E18727" if x > 0
              else SCI["colors"][2]
              for x in dfp["AUC_Diff"]]

    bars = ax.barh(range(len(dfp)), dfp["AUC_Diff"], color=colors, alpha=0.88,
                   edgecolor="white", lw=0.6, height=0.7)
    ax.set_yticks(range(len(dfp)))
    ax.set_yticklabels(dfp["Model"], fontsize=12)
    ax.set_xlabel("AUC Degradation (Internal − External)", fontsize=15)
    ax.set_title("Performance Degradation Analysis\n"
                 "(Red > 0.05 | Orange > 0 | Green ≤ 0)", fontsize=16, pad=14)
    ax.tick_params(labelsize=12)

    for bar, val in zip(bars, dfp["AUC_Diff"]):
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        offset = 0.003 if w >= 0 else -0.003
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha=ha, fontsize=11, fontweight="bold")

    ax.axvline(0, color="black", lw=1.2)
    ax.axvline(0.05, color="red", ls="--", lw=1.2, alpha=0.4, label="Warning (Δ=0.05)")
    ax.axvline(0.10, color="darkred", ls="--", lw=1.2, alpha=0.5, label="Severe (Δ=0.10)")
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    ax.grid(True, alpha=0.15, axis="x")
    sci_despine(ax)
    plt.tight_layout()
    sci_save(fig, d, "04_performance_degradation", fmts)

# ---------- M05 Radar ----------
def _plot_m05_radar(df_tr, df_ex, comp_df, d, fmts):
    sci_style()
    top6_models = comp_df.sort_values("Generalization_Score", ascending=False).head(6)["Model"].tolist()
    metrics = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1-Score"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(21, 15), subplot_kw=dict(projection="polar"))
    axes = axes.flatten()

    for i, name in enumerate(top6_models):
        if i >= len(axes):
            break
        ax = axes[i]
        tv = [df_tr.loc[name, m] for m in metrics] + [df_tr.loc[name, metrics[0]]]
        ev = [df_ex.loc[name, m] for m in metrics] + [df_ex.loc[name, metrics[0]]]

        ax.plot(angles, tv, "o-", lw=2.5, color=SCI["colors"][0],
                markersize=6, label="Internal Test")
        ax.fill(angles, tv, alpha=0.15, color=SCI["colors"][0])
        ax.plot(angles, ev, "s-", lw=2.5, color=SCI["colors"][1],
                markersize=6, label="External")
        ax.fill(angles, ev, alpha=0.15, color=SCI["colors"][1])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=11)
        ax.set_title(name, fontsize=14, fontweight="bold", pad=22)
        ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图（如果不足6个）
    for j in range(len(top6_models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Generalization Radar Chart (Internal Test vs External)",
                 fontsize=19, fontweight="bold", y=1.0)
    plt.tight_layout()
    sci_save(fig, d, "05_generalization_radar", fmts)


# ==========================================
# 执行 Step 4-6
# ==========================================

print("\n" + "=" * 70)
print("🚀 执行 Step 4-6（大字体最终版）")
print("=" * 70)

# Step 4
train_results_file = 'train_results.csv'
external_results_file = 'external_validation/external_results.csv'

comparison_df = compare_train_external_results(
    train_results_path=train_results_file,
    external_results_path=external_results_file,
    output_dir='external_validation'
)

df_train = pd.read_csv(train_results_file, index_col=0)

# 重建内部测试集
X_internal, y_internal = rebuild_internal_test(
    train_csv='train_data.csv',
    feature_names=feature_names,
    scaler=scaler,
    test_size=0.2,
    random_state=42
)
preds_internal = get_preds(models_dict, X_internal, y_internal)
preds_external = predictions

# Step 5
plot_basic_comparison_figures(
    df_train=df_train,
    df_external=df_results,
    comparison_df=comparison_df,
    trained_models=models_dict,
    preds_internal=preds_internal,
    preds_external=preds_external,
    y_internal=y_internal,
    y_external=y_external,
    output_dir='external_validation',
    fmts=("png", "pdf", "svg")
)

# Step 6
plot_medical_comparison_figures(
    df_train=df_train,
    df_external=df_results,
    comparison_df=comparison_df,
    preds_internal=preds_internal,
    preds_external=preds_external,
    y_internal=y_internal,
    y_external=y_external,
    output_dir='external_validation',
    fmts=("png", "pdf", "svg")
)

print("\n" + "=" * 70)
print("✅ Step 4-6 全部完成！（大字体最终版）")
print("=" * 70)
print("\n📁 输出:")
print("  external_validation/")
print("  ├── comparison_results.csv")
print("  ├── basic_comparison/")
print("  │   ├── 01_comparison_roc_curves.{png,pdf,svg}")
print("  │   ├── 02_comparison_metrics_bar.{png,pdf,svg}")
print("  │   ├── 03_train_vs_external_scatter.{png,pdf,svg}")
print("  │   ├── 04_comparison_calibration.{png,pdf,svg}")
print("  │   └── 05_comparison_confusion_matrix.{png,pdf,svg}")
print("  └── medical_comparison/")
print("      ├── 01_comparison_dca.{png,pdf,svg}")
print("      ├── 02_comparison_calibration_hl.{png,pdf,svg}")
print("      ├── 03_comparison_forest_plot.{png,pdf,svg}")
print("      ├── 04_performance_degradation.{png,pdf,svg}")
print("      └── 05_generalization_radar.{png,pdf,svg}")


# ==========================================
# 执行 SHAP 分析
# ==========================================




# ==========================================
# SHAP可视化分析模块 - 修复版
# ==========================================

# ==========================================
# SHAP可视化分析模块 - 基于外部验证集选模型
# ==========================================

# ==========================================
# SHAP 可解释性分析模块
# ==========================================
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

plt.rcParams['font.sans-serif'] = ['microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 函数定义（完整，不要拆开）
# ==========================================
def prepare_shap_data(trained_models, X_train, X_test, y_test,
                      feature_names,
                      external_results_csv='external_validation/external_results.csv',
                      output_dir='shap_analysis', max_samples=150):

    print("\n" + "=" * 70)
    print("🔍 SHAP 可解释性分析 - 数据准备")
    print("   📌 基于 external_results.csv 选择最优模型")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ===== 1. 从 external_results.csv 选最优模型 =====
    if not os.path.exists(external_results_csv):
        raise FileNotFoundError(f"❌ 未找到外部验证结果: {external_results_csv}")

    df_ext = pd.read_csv(external_results_csv, index_col=0)
    df_ext = df_ext.sort_values('AUC', ascending=False)

    print(f"\n📄 已加载外部验证结果: {external_results_csv}")
    print(f"   模型数: {len(df_ext)}")

    # 找外部 AUC 最高且在 trained_models 中存在的模型
    best_model_name = None
    best_ext_auc = None
    for name in df_ext.index:
        if name in trained_models:
            best_model_name = name
            best_ext_auc = df_ext.loc[name, 'AUC']
            break

    if best_model_name is None:
        raise ValueError(
            f"❌ external_results 中的模型名与 trained_models 不匹配！\n"
            f"   CSV 模型名(前5): {list(df_ext.index[:5])}\n"
            f"   已加载模型名(前5): {list(trained_models.keys())[:5]}"
        )

    best_model = trained_models[best_model_name]

    print(f"\n🏆 外部验证 AUC 最优模型: {best_model_name}")
    print(f"   External AUC: {best_ext_auc:.4f}")
    for col in ['F1-Score', 'Sensitivity', 'Specificity', 'Brier Score']:
        if col in df_ext.columns:
            print(f"   {col}: {df_ext.loc[best_model_name, col]:.4f}")

    # 打印 Top 5
    print(f"\n📊 外部验证 AUC Top 5:")
    top_cols = [c for c in ['AUC', 'Sensitivity', 'Specificity', 'F1-Score'] if c in df_ext.columns]
    print(df_ext[top_cols].head(5).to_string())

    # ===== 2. 样本抽样 =====
    if X_test.shape[0] > max_samples:
        np.random.seed(42)
        idx = np.random.choice(X_test.shape[0], max_samples, replace=False)
        X_sample = X_test[idx]
        y_sample = y_test.iloc[idx] if isinstance(y_test, pd.Series) else y_test[idx]
        print(f"\n📊 样本抽样: {X_test.shape[0]} → {max_samples}")
    else:
        X_sample = X_test
        y_sample = y_test
        print(f"\n📊 使用全部样本: {X_sample.shape[0]}")

    # ===== 3. 创建 SHAP 解释器 =====
    print("\n🧠 正在创建 SHAP 解释器...")

    shap_values = None
    base_value = None
    explainer = None
    explainer_type = "Unknown"

    try:
        # 清洗 XGBoost base_score
        if hasattr(best_model, "get_booster"):
            try:
                booster = best_model.get_booster()
                bs = booster.attr("base_score")
                if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                    bs_clean = bs.strip("[] ")
                    booster.set_attr(base_score=str(float(bs_clean)))
                    print("   ℹ️ 已修复 XGBoost base_score 格式")
            except Exception:
                pass

        # 优先 TreeExplainer
        try:
            explainer = shap.TreeExplainer(best_model)
            shap_values_raw = explainer.shap_values(X_sample)
            base_value = explainer.expected_value
            explainer_type = "TreeExplainer"
            print(f"   ✅ 使用 TreeExplainer（精确、快速）")
        except Exception as e1:
            print(f"   ⚠️ TreeExplainer 失败: {str(e1)[:60]}")
            print(f"   🔄 回退到通用 Explainer...")
            masker = shap.maskers.Independent(X_sample)
            explainer = shap.Explainer(
                lambda x: best_model.predict_proba(x)[:, 1],
                masker, feature_names=feature_names
            )
            explanation = explainer(X_sample)
            shap_values_raw = explanation.values
            base_value = explanation.base_values
            explainer_type = "KernelExplainer"
            print(f"   ✅ 使用通用 Explainer + predict_proba")

        # ===== 4. 统一 SHAP 值格式 =====
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3:
                shap_values = shap_values_raw[:, :, 1]
            elif shap_values_raw.ndim == 2:
                shap_values = shap_values_raw
            else:
                raise ValueError(f"不支持的SHAP值维度: {shap_values_raw.ndim}")
        else:
            raise ValueError(f"不支持的SHAP值类型: {type(shap_values_raw)}")

        # ===== 5. 统一 base_value 格式 =====
        if isinstance(base_value, (list, np.ndarray)):
            if hasattr(base_value, '__len__') and len(base_value) > 1:
                base_value = float(base_value[1])
            else:
                base_value = float(base_value[0]) if hasattr(base_value, '__len__') else float(base_value)
        base_value = float(base_value)

        # ===== 6. 验证形状 =====
        expected_shape = (X_sample.shape[0], len(feature_names))
        if shap_values.shape != expected_shape:
            raise ValueError(
                f"SHAP值形状不匹配！实际: {shap_values.shape}, 预期: {expected_shape}"
            )

        print(f"\n✅ SHAP 值计算完成！")
        print(f"   形状: {shap_values.shape}")
        print(f"   基准值 (Base Value): {base_value:.4f}")
        print(f"   解释器类型: {explainer_type}")

    except Exception as e:
        print(f"\n❌ SHAP 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ===== 7. 特征重要性排名 + CSV =====
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Rank': np.argsort(-mean_abs_shap).argsort() + 1
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    csv_path = os.path.join(output_dir, 'shap_feature_importance.csv')
    shap_importance.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 SHAP 特征重要性已保存: {csv_path}")
    print(f"\n🔝 Top 10 重要特征:")
    print(shap_importance.head(10).to_string(index=False))

    # ===== 8. 保存模型选择信息 =====
    model_info = pd.DataFrame([{
        'Best_Model': best_model_name,
        'External_AUC': best_ext_auc,
        'Explainer_Type': explainer_type,
        'N_Samples': X_sample.shape[0],
        'N_Features': len(feature_names),
        'Base_Value': base_value,
        'Selection_Basis': 'External Validation AUC (external_results.csv)'
    }])
    info_path = os.path.join(output_dir, 'shap_model_info.csv')
    model_info.to_csv(info_path, index=False, encoding='utf-8-sig')
    print(f"📄 模型选择信息已保存: {info_path}")

    # ===== 9. 返回 =====
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'best_ext_auc': best_ext_auc,
        'explainer': explainer,
        'explainer_type': explainer_type,
        'shap_values': shap_values,
        'base_value': base_value,
        'X_sample': X_sample,
        'y_sample': y_sample,
        'feature_names': feature_names,
        'y_pred_proba': best_model.predict_proba(X_sample)[:, 1],
        'shap_importance': shap_importance,
        'df_external_results': df_ext,
    }
# ← 函数定义到此结束（注意：上面这行 } 的右花括号后，没有缩进了）

# ==========================================
# 调用（在函数外面，顶格，无缩进）
# ==========================================
shap_data = prepare_shap_data(
    trained_models=models_dict,
    X_train=X_external,
    X_test=X_external,
    y_test=y_external,
    feature_names=feature_names,
    external_results_csv='external_validation/external_results.csv',
    output_dir='shap_analysis',
    max_samples=150
)

# ==========================================
# 2. SHAP条形图（特征重要性）
# ==========================================
# ==========================================
# SHAP 可视化模块（SCI 期刊级 + 多格式）
# ==========================================
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import os
import warnings
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

warnings.filterwarnings('ignore')

# ========== SCI 风格设置 ==========
SCI_SHAP = {
    "colors": ["#0072B5", "#BC3C29", "#20854E", "#E18727",
               "#6F99AD", "#925E9F", "#FFDC91", "#7876B1"],
    "text": "#333333", "card": "#F7F7F7", "border": "#CCCCCC",
}

def _sci_style():
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "microsoft yahei",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "axes.titlesize": 16, "axes.titleweight": "bold",
        "axes.labelsize": 14, "axes.labelweight": "bold",
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "legend.framealpha": 0.92,
        "grid.alpha": 0.25, "grid.linestyle": "--",
        "lines.linewidth": 2.2,
    })

def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _save(fig, d, name, fmts=("png", "pdf", "svg"), dpi=600):
    os.makedirs(d, exist_ok=True)
    for f in fmts:
        fig.savefig(os.path.join(d, f"{name}.{f}"), bbox_inches="tight",
                    dpi=(dpi if f in ("png", "tif") else None),
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"   ✅ {name} -> {', '.join(fmts)}")

def _title(sd, prefix=""):
    auc = sd.get('best_ext_auc', None)
    t = f"{prefix}{sd['best_model_name']}"
    if auc is not None:
        t += f"\n(External Validation AUC = {auc:.3f})"
    return t

# ==========================================
# 01. SHAP 特征重要性条形图
# ==========================================
def plot_shap_bar_importance(shap_data, output_dir='shap_analysis',
                             fmts=("png", "pdf", "svg")):
    print("\n📊 1. SHAP 特征重要性条形图...")
    _sci_style()

    sv = shap_data['shap_values']
    fn = shap_data['feature_names']

    mean_abs = np.abs(sv).mean(axis=0)
    total = mean_abs.sum()
    imp = pd.DataFrame({'Feature': fn, 'Importance': mean_abs}
                       ).sort_values('Importance', ascending=True)
    if len(imp) > 20:
        imp = imp.tail(20)

    fig, ax = plt.subplots(figsize=(14, 10))
    norm = Normalize(vmin=imp['Importance'].min(), vmax=imp['Importance'].max())
    colors = [plt.cm.Blues(0.3 + 0.65 * norm(v)) for v in imp['Importance']]

    bars = ax.barh(range(len(imp)), imp['Importance'], color=colors,
                   edgecolor="white", lw=0.6, height=0.72)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp['Feature'], fontsize=13)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=15)
    ax.set_title(_title(shap_data, "SHAP Feature Importance — "),
                 fontsize=17, fontweight="bold", pad=16)
    ax.tick_params(labelsize=13)

    mx = imp['Importance'].max()
    for bar, val in zip(bars, imp['Importance']):
        pct = val / total * 100
        ax.text(val + mx * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ({pct:.1f}%)", va="center", fontsize=11,
                fontweight="bold", color=SCI_SHAP["text"])

    ax.text(0.02, -0.06,
            f"Based on {sv.shape[0]} samples | "
            f"Explainer: {shap_data.get('explainer_type', 'N/A')}",
            transform=ax.transAxes, fontsize=10, color="#999")
    ax.grid(True, axis="x", alpha=0.15)
    _despine(ax)
    plt.tight_layout()
    _save(fig, output_dir, "01_shap_feature_importance_bar", fmts)

# ==========================================
# 02. SHAP 蜂群图
# ==========================================
def plot_shap_summary_beeswarm(shap_data, output_dir='shap_analysis',
                                fmts=("png", "pdf", "svg")):
    print("\n📊 2. SHAP 蜂群图...")
    _sci_style()

    fig = plt.figure(figsize=(14, 11))
    X_df = pd.DataFrame(shap_data['X_sample'], columns=shap_data['feature_names'])
    shap.summary_plot(shap_data['shap_values'], X_df,
                      plot_type="dot", max_display=20, show=False, plot_size=None)

    ax = plt.gca()
    ax.set_title(_title(shap_data, "SHAP Summary — "),
                 fontsize=17, fontweight="bold", pad=14)
    ax.set_xlabel("SHAP Value (→ Higher Risk | ← Lower Risk)", fontsize=14)
    ax.tick_params(labelsize=12)
    fig = plt.gcf()
    fig.set_size_inches(14, 11)
    plt.tight_layout()
    _save(fig, output_dir, "02_shap_summary_beeswarm", fmts)

# ==========================================
# 03. SHAP 统计报告
# ==========================================
def generate_shap_statistics(shap_data, output_dir='shap_analysis'):
    print("\n📊 3. SHAP 统计报告...")
    os.makedirs(output_dir, exist_ok=True)

    sv = shap_data['shap_values']
    fn = shap_data['feature_names']

    df = pd.DataFrame({
        'Feature': fn,
        'Mean_|SHAP|': np.abs(sv).mean(0),
        'Std_SHAP': sv.std(0),
        'Max_|SHAP|': np.abs(sv).max(0),
        'Min_|SHAP|': np.abs(sv).min(0),
        'Mean_SHAP': sv.mean(0),
        'Median_SHAP': np.median(sv, axis=0),
        'Pct_Positive': (sv > 0).mean(0) * 100,
    }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)

    df['Rank'] = range(1, len(df) + 1)
    total = df['Mean_|SHAP|'].sum()
    df['Contribution_%'] = (df['Mean_|SHAP|'] / total * 100).round(2)
    df['Cumulative_%'] = df['Contribution_%'].cumsum().round(2)

    p = os.path.join(output_dir, 'shap_statistics_report.csv')
    df.to_csv(p, index=False, encoding='utf-8-sig')
    print(f"   ✅ {p}")
    print(f"\n🔝 Top 15:")
    print(df.head(15).to_string(index=False))
    return df
# ==========================================
# 04. SHAP 瀑布图（典型样本）
# ==========================================
def plot_shap_waterfall_samples(shap_data, output_dir='shap_analysis',
                                 n_samples=5, fmts=("png", "pdf", "svg")):
    print(f"\n📊 4. SHAP 瀑布图 ({n_samples} 样本)...")
    _sci_style()

    sv = shap_data['shap_values']
    bv = shap_data['base_value']
    xs = shap_data['X_sample']
    fn = shap_data['feature_names']
    yp = shap_data['y_pred_proba']
    ys = shap_data['y_sample']

    exp = shap.Explanation(values=sv, base_values=bv, data=xs, feature_names=fn)

    indices = []
    indices.append((np.argmax(yp), 'Highest_Risk'))
    indices.append((np.argmin(yp), 'Lowest_Risk'))
    indices.append((np.argmin(np.abs(yp - 0.5)), 'Borderline'))

    remain = min(n_samples - 3, len(xs))
    if remain > 0:
        np.random.seed(42)
        for i, idx in enumerate(np.random.choice(len(xs), remain, replace=False)):
            indices.append((idx, f'Random_{i + 1}'))

    for si, label in indices[:n_samples]:
        fig = plt.figure(figsize=(13, 9))
        shap.plots.waterfall(exp[si], max_display=15, show=False)

        ax = plt.gca()
        prob = yp[si]
        actual = ys.iloc[si] if isinstance(ys, pd.Series) else ys[si]
        pred_cls = "Positive" if prob > 0.5 else "Negative"
        act_cls = "Positive" if actual == 1 else "Negative"

        ax.set_title(
            f"Waterfall — {label}\n"
            f"Pred: {prob:.3f} ({pred_cls}) | Actual: {act_cls}\n"
            f"Model: {shap_data['best_model_name']}",
            fontsize=15, fontweight="bold", pad=14)
        ax.tick_params(labelsize=12)
        fig = plt.gcf(); fig.set_size_inches(13, 9)
        plt.tight_layout()
        _save(fig, output_dir, f"03_shap_waterfall_{label}", fmts)

# ==========================================
# 05. SHAP 依赖图（Top4）
# ==========================================
def plot_shap_dependence(shap_data, output_dir='shap_analysis',
                          top_n=4, fmts=("png", "pdf", "svg")):
    print(f"\n📊 5. SHAP 依赖图 (Top {top_n})...")
    _sci_style()

    sv = shap_data['shap_values']
    xs = shap_data['X_sample']
    fn = shap_data['feature_names']
    yp = shap_data['y_pred_proba']

    imp = np.abs(sv).mean(0)
    top_idx = np.argsort(imp)[::-1][:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for i, fi in enumerate(top_idx):
        ax = axes[i]
        xv = xs[:, fi]
        yv = sv[:, fi]

        sc = ax.scatter(xv, yv, c=yp, cmap="RdYlBu_r", s=50, alpha=0.75,
                        edgecolors="gray", lw=0.3, zorder=3)
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.4)

        # 趋势线 + 相关系数
        try:
            z = np.polyfit(xv, yv, 1)
            xl = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), color=SCI_SHAP["colors"][1],
                    ls="--", lw=2, alpha=0.7)
            r = np.corrcoef(xv, yv)[0, 1]
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=12, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="#CCC"))
        except Exception:
            pass

        ax.set_xlabel(f"{fn[fi]} (Feature Value)", fontsize=13)
        ax.set_ylabel("SHAP Value", fontsize=13)
        ax.set_title(f"#{i + 1}  {fn[fi]}", fontsize=14, fontweight="bold", pad=10)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.15)
        _despine(ax)

    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label("Predicted Probability", fontsize=12)
    cb.ax.tick_params(labelsize=11)

    fig.suptitle(_title(shap_data, f"SHAP Dependence — Top {top_n} Features\n"),
                 fontsize=17, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    _save(fig, output_dir, f"04_shap_dependence_top{top_n}", fmts)

# ==========================================
# 06. SHAP Force Plot（静态）
# ==========================================
def plot_shap_force_plots(shap_data, output_dir='shap_analysis',
                           n_samples=3, fmts=("png", "pdf", "svg")):
    print(f"\n📊 6. Force Plot ({n_samples} 样本)...")
    _sci_style()
    os.makedirs(output_dir, exist_ok=True)

    sv = np.round(shap_data['shap_values'], 2)
    xv = np.round(shap_data['X_sample'], 2)
    bv = shap_data['base_value']
    fn = shap_data['feature_names']
    yp = shap_data['y_pred_proba']
    ys = shap_data['y_sample']
    model_name = shap_data['best_model_name']

    # 选择有代表性的样本：最高风险 + 最低风险 + 随机
    selected = []
    selected.append((np.argmax(yp), "HighRisk"))
    selected.append((np.argmin(yp), "LowRisk"))

    np.random.seed(42)
    remain = min(n_samples - 2, len(xv) - 2)
    if remain > 0:
        pool = [i for i in range(len(xv)) if i not in [selected[0][0], selected[1][0]]]
        for i, idx in enumerate(np.random.choice(pool, remain, replace=False)):
            selected.append((idx, f"Random{i + 1}"))

    for idx, label in selected[:n_samples]:
        # 关键：先关闭所有旧图，让 shap 创建新图
        plt.close('all')

        # 调用 shap.force_plot（它会自己创建 figure）
        shap.force_plot(
            bv, sv[idx], xv[idx],
            feature_names=fn,
            matplotlib=True,
            show=False,
            text_rotation=20
        )

        # 获取 shap 创建的 figure
        fig = plt.gcf()
        fig.set_size_inches(22, 6)

        # 添加标题（用 fig.text 而不是 ax.set_title，避免被遮挡）
        actual = ys.iloc[idx] if isinstance(ys, pd.Series) else ys[idx]
        pred_cls = "High Risk" if yp[idx] >= 0.5 else "Low Risk"
        act_cls = "Positive" if actual == 1 else "Negative"

        fig.text(
            0.5, 0.98,
            f"Force Plot — {label} (Sample #{idx})\n"
            f"Pred: {yp[idx]:.3f} ({pred_cls}) | Actual: {act_cls} | Model: {model_name}",
            ha="center", va="top", fontsize=14, fontweight="bold"
        )

        # 调整布局，给标题留空间
        fig.subplots_adjust(top=0.75, bottom=0.15, left=0.05, right=0.95)

        # 保存
        _save(fig, output_dir, f"05_shap_force_{label}_{idx}", fmts)

    print(f"   共生成 {len(selected[:n_samples])} 张 Force Plot")

# ==========================================
# 07. SHAP 决策路径图
# ==========================================
def plot_shap_decision(shap_data, output_dir='shap_analysis',
                        n_samples=100, fmts=("png", "pdf", "svg")):
    print(f"\n📊 7. 决策路径图 ({n_samples} 样本)...")
    _sci_style()

    sv = shap_data['shap_values']
    bv = shap_data['base_value']
    fn = shap_data['feature_names']
    xs = shap_data['X_sample']
    yp = shap_data['y_pred_proba']

    n = min(n_samples, len(xs))
    np.random.seed(42)
    ids = np.random.choice(len(xs), n, replace=False)

    X_df = pd.DataFrame(xs, columns=fn)

    fig = plt.figure(figsize=(13, 9))
    shap.decision_plot(bv, sv[ids], X_df.iloc[ids],
                       feature_names=fn, show=False, link='logit',
                       highlight=np.where(yp[ids] > 0.5)[0])

    ax = plt.gca()
    ax.set_title(_title(shap_data, f"Decision Plot ({n} samples)\n"),
                 fontsize=15, fontweight="bold", pad=14)
    ax.tick_params(labelsize=12)
    fig = plt.gcf(); fig.set_size_inches(13, 9)
    plt.tight_layout()
    _save(fig, output_dir, "06_shap_decision_plot", fmts)

# ==========================================
# 08. SHAP 热力图
# ==========================================
def plot_shap_heatmap(shap_data, output_dir='shap_analysis',
                       fmts=("png", "pdf", "svg")):
    print("\n📊 8. SHAP 热力图...")
    _sci_style()

    sv = shap_data['shap_values']
    fn = shap_data['feature_names']
    ys = shap_data['y_sample']

    imp = np.abs(sv).mean(0)
    top15 = np.argsort(imp)[::-1][:15]
    shap_sum = sv.sum(1)
    sorted_idx = np.argsort(shap_sum)

    n_show = min(60, len(sorted_idx))
    sel = sorted_idx[-n_show:]
    hm_data = sv[sel][:, top15].T

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(hm_data,
                yticklabels=[fn[i] for i in top15], xticklabels=[],
                cmap="RdBu_r", center=0, ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "SHAP Value", "shrink": 0.8})
    ax.tick_params(labelsize=12)

    # 底部标签色条
    ys_arr = ys.values if isinstance(ys, pd.Series) else ys
    labels = ys_arr[sel]
    for i, lb in enumerate(labels):
        ax.axvline(x=i + 0.5, color=SCI_SHAP["colors"][1] if lb == 1 else SCI_SHAP["colors"][0],
                   alpha=0.7, lw=2.5, ymin=0, ymax=0.02)

    legend_elements = [
        Patch(facecolor=SCI_SHAP["colors"][1], label="Positive (1)"),
        Patch(facecolor=SCI_SHAP["colors"][0], label="Negative (0)")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11, frameon=True,
              title="True Label", title_fontsize=12)

    ax.set_xlabel("Samples (sorted by SHAP sum)", fontsize=14)
    ax.set_ylabel("Features (sorted by importance)", fontsize=14)
    ax.set_title(_title(shap_data, "SHAP Heatmap — "),
                 fontsize=16, fontweight="bold", pad=14)
    plt.tight_layout()
    _save(fig, output_dir, "07_shap_heatmap", fmts)

# ==========================================
# 09. SHAP 综合看板
# ==========================================
def plot_shap_comprehensive_dashboard(shap_data, output_dir='shap_analysis',
                                       fmts=("png", "pdf", "svg")):
    print("\n📊 9. SHAP 综合看板...")
    _sci_style()

    sv = shap_data['shap_values']
    xs = shap_data['X_sample']
    fn = shap_data['feature_names']
    ys = shap_data['y_sample']
    ys_arr = ys.values if isinstance(ys, pd.Series) else ys

    imp = np.abs(sv).mean(0)
    rank = np.argsort(imp)[::-1]

    # ===== 缩小画布（原来 26×17 太大） =====
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, wspace=0.45, hspace=0.45)

    # 左侧：手动蜂群图
    ax_main = fig.add_subplot(gs[:, :2])
    top20 = rank[:20]
    top20_names = [fn[i] for i in top20]

    ax_bg = ax_main.twiny()
    ax_bg.barh(range(20), imp[top20][::-1], color="#E8E8E8", height=0.7, zorder=0)
    ax_bg.set_xlabel("Mean |SHAP|", fontsize=12, color="#999")
    ax_bg.tick_params(labelsize=10, colors="#999")

    cmap = plt.get_cmap("RdYlBu_r")
    for i, fi in enumerate(top20[::-1]):
        sv_f = sv[:, fi]
        fv_f = xs[:, fi]
        fv_norm = (fv_f - fv_f.min()) / (fv_f.max() - fv_f.min() + 1e-10)
        jitter = np.random.normal(0, 0.08, len(sv_f))
        ax_main.scatter(sv_f, i + jitter, c=cmap(fv_norm), s=18, alpha=0.75, edgecolor="none")

    ax_main.set_yticks(range(20))
    ax_main.set_yticklabels(top20_names[::-1], fontsize=11)
    ax_main.set_xlabel("SHAP Value", fontsize=13, fontweight="bold")
    ax_main.set_title("Global Feature Importance (Beeswarm)", fontsize=14, fontweight="bold")
    ax_main.grid(True, axis="x", alpha=0.2)
    ax_main.axvline(0, color="black", lw=0.8, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    cax = fig.add_axes([0.12, 0.06, 0.28, 0.012])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Feature Value (Low → High)", fontsize=11)

    # 右侧：Top6 依赖图
    top6 = rank[:6]
    for k in range(6):
        r, c = divmod(k, 2)
        ax = fig.add_subplot(gs[r, c + 2])
        fi = top6[k]
        xv, yv = xs[:, fi], sv[:, fi]

        sc = ax.scatter(xv, yv, c=ys_arr, cmap="coolwarm", s=35, alpha=0.7,
                        edgecolor="k", lw=0.2)
        ax.axhline(0, color="black", ls="--", lw=0.7, alpha=0.4)

        try:
            z = np.polyfit(xv, yv, 1)
            xl = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), "k--", lw=1.5, alpha=0.6)
            r_val = np.corrcoef(xv, yv)[0, 1]
            ax.text(0.05, 0.95, f"r={r_val:.2f}", transform=ax.transAxes,
                    fontsize=10, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.9))
        except Exception:
            pass

        ax.set_xlabel(fn[fi], fontsize=11)
        ax.set_ylabel("SHAP", fontsize=11)
        ax.set_title(f"Top {k + 1}: {fn[fi]}", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.15)
        _despine(ax)

    cax2 = fig.add_axes([0.93, 0.38, 0.01, 0.22])
    cb2 = fig.colorbar(sc, cax=cax2)
    cb2.set_label("True Label", fontsize=11)
    cb2.set_ticks([0, 1])
    cb2.set_ticklabels(["Neg", "Pos"])

    fig.suptitle(_title(shap_data, "SHAP Comprehensive Dashboard — "),
                 fontsize=20, fontweight="bold", y=0.97)

    # ===== 关键修改：降低 DPI 防止超大图报错 =====
    _save(fig, output_dir, "08_shap_comprehensive_dashboard", fmts, dpi=300)

# ==========================================
# 10. LIME 对比分析
# ==========================================
def plot_lime_analysis(shap_data, X_train, output_dir='shap_analysis',
                        fmts=("png", "pdf", "svg")):
    print("\n📊 10. LIME 对比分析...")
    _sci_style()

    model = shap_data['best_model']
    xs = shap_data['X_sample']
    ys = shap_data['y_sample']
    fn = shap_data['feature_names']
    yp = shap_data['y_pred_proba']
    sv = shap_data['shap_values']

    np.random.seed(42)
    bg = X_train[np.random.choice(X_train.shape[0], min(5000, X_train.shape[0]), replace=False)]

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data=bg, feature_names=fn,
        class_names=["Low Risk", "High Risk"],
        mode="classification", discretize_continuous=True, random_state=42
    )

    hi = np.argmax(yp)
    exp = lime_exp.explain_instance(xs[hi], model.predict_proba, num_features=10)
    exp_list = exp.as_list()

    # --- 图 1：LIME 单独 ---
    feats = [x[0] for x in exp_list]
    wts = [x[1] for x in exp_list]
    colors = [SCI_SHAP["colors"][1] if w > 0 else SCI_SHAP["colors"][0] for w in wts]

    fig, ax = plt.subplots(figsize=(13, 8))
    bars = ax.barh(range(len(wts)), wts, color=colors, alpha=0.85,
                   edgecolor="white", lw=0.5, height=0.65)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=12)
    ax.invert_yaxis()

    for bar, w in zip(bars, wts):
        ax.text(w + (0.003 if w >= 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=11, fontweight="bold",
                ha="left" if w >= 0 else "right")

    ys_arr = ys.values if isinstance(ys, pd.Series) else ys
    true_lb = "Positive" if ys_arr[hi] == 1 else "Negative"
    pred_cls = "High Risk" if yp[hi] >= 0.5 else "Low Risk"
    ax.set_title(f"LIME — High-Risk Sample #{hi}\n"
                 f"Pred: {yp[hi]:.3f} ({pred_cls}) | True: {true_lb}",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Feature Contribution (Weight)", fontsize=14)
    ax.axvline(0, color="black", lw=1)
    ax.grid(True, axis="x", alpha=0.15)
    _despine(ax)

    legend_elements = [
        Patch(facecolor=SCI_SHAP["colors"][1], label="Risk ↑"),
        Patch(facecolor=SCI_SHAP["colors"][0], label="Risk ↓")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=12)
    plt.tight_layout()
    _save(fig, output_dir, "09_lime_high_risk_sample", fmts)

    # --- 图 2：LIME vs SHAP ---
    comp = []
    for lname, lw in exp_list:
        for raw in fn:
            if raw in lname:
                comp.append({
                    "Feature": raw,
                    "LIME_Weight": lw,
                    "SHAP_Value": sv[hi, fn.index(raw)]
                })
                break

    if not comp:
        print("   ⚠️ LIME 特征无法匹配，跳过对比图")
        return

    cdf = pd.DataFrame(comp)
    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(cdf))
    h = 0.35

    b1 = ax.barh(y_pos + h / 2, cdf["LIME_Weight"], h,
                 label="LIME Weight", color=SCI_SHAP["colors"][2], alpha=0.85,
                 edgecolor="white", lw=0.5)
    b2 = ax.barh(y_pos - h / 2, cdf["SHAP_Value"], h,
                 label="SHAP Value", color=SCI_SHAP["colors"][3], alpha=0.85,
                 edgecolor="white", lw=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cdf["Feature"], fontsize=12)
    ax.invert_yaxis()
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Contribution Value", fontsize=14)
    ax.set_title(f"LIME vs SHAP — Sample #{hi}\n"
                 f"Model: {shap_data['best_model_name']}",
                 fontsize=15, fontweight="bold", pad=14)
    ax.legend(fontsize=12, loc="lower right")
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="x", alpha=0.15)
    _despine(ax)

    for bars, vals in [(b1, cdf["LIME_Weight"]), (b2, cdf["SHAP_Value"])]:
        for bar, v in zip(bars, vals):
            w = bar.get_width()
            ax.text(w + (0.002 if w >= 0 else -0.002),
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=10,
                    ha="left" if w >= 0 else "right")

    corr = cdf["LIME_Weight"].corr(cdf["SHAP_Value"])
    ax.text(0.02, -0.07, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9, ec="#CCC"))
    plt.tight_layout()
    _save(fig, output_dir, "10_lime_vs_shap_comparison", fmts)

# ==========================================
# 主调用函数
# ==========================================
def run_complete_shap_analysis(shap_data, X_train=None, output_dir='shap_analysis',
                                fmts=("png", "pdf", "svg")):
    if shap_data is None:
        print("❌ shap_data 为空")
        return

    print("\n" + "=" * 70)
    print(f"🎨 完整 SHAP 可视化 — {shap_data['best_model_name']}")
    auc = shap_data.get('best_ext_auc', None)
    if auc:
        print(f"   External AUC = {auc:.4f}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    plot_shap_bar_importance(shap_data, output_dir, fmts)
    plot_shap_summary_beeswarm(shap_data, output_dir, fmts)
    generate_shap_statistics(shap_data, output_dir)
    plot_shap_waterfall_samples(shap_data, output_dir, n_samples=5, fmts=fmts)
    plot_shap_dependence(shap_data, output_dir, top_n=4, fmts=fmts)
    plot_shap_force_plots(shap_data, output_dir, n_samples=3, fmts=fmts)
    plot_shap_decision(shap_data, output_dir, n_samples=100, fmts=fmts)
    plot_shap_heatmap(shap_data, output_dir, fmts)
    plot_shap_comprehensive_dashboard(shap_data, output_dir, fmts)

    if X_train is not None:
        plot_lime_analysis(shap_data, X_train, output_dir, fmts)
    else:
        print("\n⚠️ 未提供 X_train，跳过 LIME 分析")

    print(f"\n✅ 全部完成！→ {output_dir}/")

# ==========================================
# 执行
# ==========================================
print("\n" + "=" * 70)
print("🚀 开始完整 SHAP 可解释性分析")
print("=" * 70)

run_complete_shap_analysis(
    shap_data=shap_data,
    X_train=X_external,
    output_dir='shap_analysis',
    fmts=("png", "pdf", "svg")
)

print("\n📁 shap_analysis/")
print("  📄 shap_statistics_report.csv")
print("  📄 shap_feature_importance.csv")
print("  📄 shap_model_info.csv")
print("  🖼️ 01_shap_feature_importance_bar.{png,pdf,svg}")
print("  🖼️ 02_shap_summary_beeswarm.{png,pdf,svg}")
print("  🖼️ 03_shap_waterfall_*.{png,pdf,svg}  (×5)")
print("  🖼️ 04_shap_dependence_top4.{png,pdf,svg}")
print("  🖼️ 05_shap_force_sample_*.{png,pdf,svg}  (×3)")
print("  🖼️ 06_shap_decision_plot.{png,pdf,svg}")
print("  🖼️ 07_shap_heatmap.{png,pdf,svg}")
print("  🖼️ 08_shap_comprehensive_dashboard.{png,pdf,svg}")
print("  🖼️ 09_lime_high_risk_sample.{png,pdf,svg}")
print("  🖼️ 10_lime_vs_shap_comparison.{png,pdf,svg}")




# ==========================================
# 11. 交互式 SHAP Force Plot (HTML)
# ==========================================
def plot_shap_force_interactive_html(shap_data, output_dir='shap_analysis'):
    """生成交互式 SHAP Force Plot（HTML 格式，浏览器打开）"""
    print("\n📊 11. 交互式 Force Plot (HTML)...")

    sv = shap_data['shap_values']
    bv = shap_data['base_value']
    xs = shap_data['X_sample']
    fn = shap_data['feature_names']

    os.makedirs(output_dir, exist_ok=True)

    try:
        X_df = pd.DataFrame(xs, columns=fn)

        interactive_plot = shap.force_plot(
            bv, sv, X_df, feature_names=fn
        )

        html_path = os.path.join(output_dir, '11_shap_force_interactive.html')
        shap.save_html(html_path, interactive_plot)

        print(f"   ✅ {html_path}")
        print(f"   → 浏览器打开可交互：排序、聚类、悬停查看详情")

    except Exception as e:
        print(f"   ⚠️ HTML 保存失败: {e}")
        print(f"   → 建议: pip install --upgrade shap")

# ==========================================
# 主函数：执行完整 SHAP 分析流程
# ==========================================
def run_complete_shap_analysis(shap_data, X_train=None,
                                output_dir='shap_analysis',
                                fmts=("png", "pdf", "svg")):
    """
    执行完整 SHAP + LIME 可解释性分析

    参数:
        shap_data: prepare_shap_data() 返回的字典
        X_train: 背景数据（LIME 需要，可选）
        output_dir: 输出目录
        fmts: 图片保存格式
    """
    if shap_data is None:
        print("❌ shap_data 为空，终止分析")
        return

    print("\n" + "=" * 70)
    print(f"🎨 完整 SHAP 可视化 — {shap_data['best_model_name']}")
    auc = shap_data.get('best_ext_auc', None)
    if auc is not None:
        print(f"   External Validation AUC = {auc:.4f}")
    print(f"   样本数: {shap_data['shap_values'].shape[0]}")
    print(f"   特征数: {shap_data['shap_values'].shape[1]}")
    print(f"   解释器: {shap_data.get('explainer_type', 'N/A')}")
    print(f"   输出格式: {', '.join(fmts)}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. 特征重要性条形图
    plot_shap_bar_importance(shap_data, output_dir, fmts)

    # 2. 蜂群图
    plot_shap_summary_beeswarm(shap_data, output_dir, fmts)

    # 3. 统计报告
    generate_shap_statistics(shap_data, output_dir)

    # 4. 瀑布图（5 个典型样本）
    plot_shap_waterfall_samples(shap_data, output_dir, n_samples=5, fmts=fmts)

    # 5. 依赖图（Top4）
    plot_shap_dependence(shap_data, output_dir, top_n=4, fmts=fmts)

    # 6. 力图（3 个样本）
    plot_shap_force_plots(shap_data, output_dir, n_samples=3, fmts=fmts)

    # 7. 决策路径图
    plot_shap_decision(shap_data, output_dir, n_samples=100, fmts=fmts)

    # 8. 热力图
    plot_shap_heatmap(shap_data, output_dir, fmts)

    # 9. 综合看板
    plot_shap_comprehensive_dashboard(shap_data, output_dir, fmts)

    # 10. LIME 对比分析
    if X_train is not None:
        try:
            plot_lime_analysis(shap_data, X_train, output_dir, fmts)
        except Exception as e:
            print(f"\n⚠️ LIME 分析失败（可跳过）: {e}")
    else:
        print("\n⚠️ 未提供 X_train，跳过 LIME 分析")

    # 11. 交互式 HTML
    plot_shap_force_interactive_html(shap_data, output_dir)

    # 打印汇总
    print("\n" + "=" * 70)
    print("🎉 所有 SHAP 分析完成！")
    print("=" * 70)
    print(f"\n📁 {output_dir}/")
    print(f"  📄 shap_feature_importance.csv")
    print(f"  📄 shap_statistics_report.csv")
    print(f"  📄 shap_model_info.csv")
    print(f"  🖼️ 01_shap_feature_importance_bar.{{{','.join(fmts)}}}")
    print(f"  🖼️ 02_shap_summary_beeswarm.{{{','.join(fmts)}}}")
    print(f"  🖼️ 03_shap_waterfall_*.{{{','.join(fmts)}}}  (×5)")
    print(f"  🖼️ 04_shap_dependence_top4.{{{','.join(fmts)}}}")
    print(f"  🖼️ 05_shap_force_sample_*.{{{','.join(fmts)}}}  (×3)")
    print(f"  🖼️ 06_shap_decision_plot.{{{','.join(fmts)}}}")
    print(f"  🖼️ 07_shap_heatmap.{{{','.join(fmts)}}}")
    print(f"  🖼️ 08_shap_comprehensive_dashboard.{{{','.join(fmts)}}}")
    print(f"  🖼️ 09_lime_high_risk_sample.{{{','.join(fmts)}}}")
    print(f"  🖼️ 10_lime_vs_shap_comparison.{{{','.join(fmts)}}}")
    print(f"  🌐 11_shap_force_interactive.html")

# ==========================================
# 执行
# ==========================================
print("\n" + "=" * 70)
print("🚀 开始完整 SHAP 可解释性分析")
print("=" * 70)

run_complete_shap_analysis(
    shap_data=shap_data,          # prepare_shap_data() 的返回值
    X_train=X_external,           # 背景数据（LIME 需要）
    output_dir='shap_analysis',
    fmts=("png", "pdf", "svg")
)

print("\n" + "=" * 70)
print("✅ 所有分析和可视化已完成！")
print("=" * 70)
