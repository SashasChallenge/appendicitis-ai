import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier  # 新增
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import optuna  # 新增
from optuna.samplers import TPESampler
import joblib
import os
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['microsoft yahei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 数据准备与分割
# ==========================================
def prepare_data(file_path):
    """加载清洗后的数据并准备特征/标签"""
    print("=" * 60)
    print("Step 1: 数据准备与分割")
    print("=" * 60)

    # 尝试多种编码方式读取
    encodings = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin1', 'cp1252']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✓ 成功读取文件，编码: {encoding}")
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if df is None:
        raise ValueError(f"无法读取文件 {file_path}，尝试的编码: {encodings}")

    print(f"数据形状: {df.shape}")

    X = df.drop(columns=['Target'])


    y = df['Target']

    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"删除非数值列: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)

    print(f"特征数量: {X.shape[1]}")
    print(f"正样本比例: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ==========================================
# 处理类别不平衡
# ==========================================

def handle_imbalance(X_train, y_train, method='smote'):
    """处理类别不平衡"""
    print("\n" + "=" * 60)
    print("Step 2: 处理类别不平衡")
    print("=" * 60)

    print(f"处理前 - 正样本: {y_train.sum()}, 负样本: {len(y_train) - y_train.sum()}")

    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"SMOTE后 - 正样本: {y_resampled.sum()}, 负样本: {len(y_resampled) - y_resampled.sum()}")
        return X_resampled, y_resampled
    else:
        return X_train, y_train


# ==========================================
# 特征标准化
# ==========================================

def scale_features(X_train, X_test):
    """特征标准化"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ==========================================
# Optuna超参数优化
# ==========================================

def optimize_xgboost(X_train, y_train, n_trials=50):
    """使用Optuna优化XGBoost"""
    print("\n" + "=" * 60)
    print("Optuna优化 XGBoost")
    print("=" * 60)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'verbosity': 0
        }

        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"最佳AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")

    best_model = xgb.XGBClassifier(**study.best_params)
    return best_model, study.best_params


def optimize_lightgbm(X_train, y_train, n_trials=50):
    """使用Optuna优化LightGBM"""
    print("\n" + "=" * 60)
    print("Optuna优化 LightGBM")
    print("=" * 60)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1
        }

        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"最佳AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")

    best_model = lgb.LGBMClassifier(**study.best_params)
    return best_model, study.best_params


def optimize_random_forest(X_train, y_train, n_trials=50):
    """使用Optuna优化Random Forest"""
    print("\n" + "=" * 60)
    print("Optuna优化 Random Forest")
    print("=" * 60)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }

        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"最佳AUC: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")

    best_model = RandomForestClassifier(**study.best_params)
    return best_model, study.best_params


# ==========================================
# 模型定义
# ==========================================

def define_models():
    """定义11个基础模型"""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=10,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski',
            n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Naive Bayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    return models


def create_ensemble_models(base_models, optimized_models):
    """创建Stacking和Voting集成模型"""
    print("\n" + "=" * 60)
    print("创建集成模型")
    print("=" * 60)

    ensemble_models = {}

    voting_estimators = [
        ('xgb_opt', optimized_models['XGBoost Optimized']),
        ('lgb_opt', optimized_models['LightGBM Optimized']),
        ('rf_opt', optimized_models['Random Forest Optimized']),
        ('lr', base_models['Logistic Regression']),
        ('et', base_models['Extra Trees'])
    ]

    # 只保留Soft Voting（支持概率）
    voting_soft = VotingClassifier(
        estimators=voting_estimators,
        voting='soft',
        n_jobs=-1
    )
    ensemble_models['Voting Soft'] = voting_soft
    print("✓ 创建 Voting (Soft) 模型")

    # Stacking Classifier
    stacking = StackingClassifier(
        estimators=voting_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )
    ensemble_models['Stacking'] = stacking
    print("✓ 创建 Stacking 模型")

    return ensemble_models

from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
"""
分层特征选择策略。
📊 特征选择方法建议与理由
我们采用 "过滤法 (Filter) + 嵌入法 (Embedded)" 的组合策略：
方差阈值过滤 (Variance Threshold)
理由：最基础的清洗。剔除那些所有样本数值都一样的特征（方差为0），这些特征对分类没有任何贡献。
相关性分析 (Correlation Analysis)
理由：医学指标中常存在高度相关的特征（如：白蛋白与总蛋白）。高共线性会干扰线性模型（如逻辑回归）的权重稳定性。我们移除相关系数 > 0.9 的冗余特征（保留与目标变量相关性更高的那个）。
基于随机森林的递归特征消除 (RFE - Recursive Feature Elimination)
"""


# ==========================================
# 模型训练与评估
# ==========================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, models_dict):
    """训练并评估所有模型"""
    print("\n" + "=" * 60)
    print(f"Step 3: 模型训练与评估 ({len(models_dict)}个模型)")
    print("=" * 60)

    results = {}
    trained_models = {}
    predictions = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models_dict.items():
        print(f"\n{'─' * 40}")
        print(f"训练模型: {name}")
        print(f"{'─' * 40}")

        try:
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            print(f"5折CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # 训练模型
            model.fit(X_train, y_train)
            trained_models[name] = model

            # 预测
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            predictions[name] = {
                'y_pred': y_pred,
                'y_prob': y_prob
            }

            # 计算评估指标
            auc = roc_auc_score(y_test, y_prob)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            brier = brier_score_loss(y_test, y_prob)
            mcc = matthews_corrcoef(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)
            ap = average_precision_score(y_test, y_prob)

            youden = sensitivity + specificity - 1

            results[name] = {
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
                'CV_AUC_mean': cv_scores.mean(),
                'CV_AUC_std': cv_scores.std(),
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            }

            print(f"测试集 AUC: {auc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")

        except Exception as e:
            print(f"模型 {name} 训练失败: {str(e)}")
            continue

    return results, trained_models, predictions


# ==========================================
# 可视化函数
# ==========================================

def plot_roc_curves(trained_models, X_test, y_test, filename='train_roc_curves.png'):
    """绘制ROC曲线"""
    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(trained_models)))

    for (name, model), color in zip(trained_models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=14)
    plt.ylabel('Sensitivity (True Positive Rate)', fontsize=14)
    plt.title('ROC Curves Comparison - All Models', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存: {filename}")


def plot_metrics_heatmap(results, filename='train_metrics_heatmap.png'):
    """绘制指标热力图"""
    metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
               'Sensitivity', 'Specificity', 'PPV', 'NPV', 'MCC', 'Kappa', 'AP']

    df_results = pd.DataFrame(results).T
    df_plot = df_results[metrics].sort_values('AUC', ascending=False)

    plt.figure(figsize=(16, 12))
    sns.heatmap(df_plot, annot=True, fmt='.3f', cmap='RdYlGn',
                linewidths=0.5, center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Score'})

    plt.title('Model Performance Heatmap - All Models', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Models', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"指标热力图已保存: {filename}")


def plot_model_comparison_bar(results, filename='model_comparison_bar.png'):
    """绘制模型对比柱状图"""
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('AUC', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    metrics = ['AUC', 'F1-Score', 'Sensitivity', 'Specificity']
    x = np.arange(len(df_results))
    width = 0.2

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.barh(x + i * width, df_results[metric], width, label=metric, color=color, alpha=0.8)
        for bar, val in zip(bars, df_results[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8)

    ax.set_yticks(x + width * 1.5)
    ax.set_yticklabels(df_results.index, fontsize=10)
    ax.set_xlabel('Score', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型对比柱状图已保存: {filename}")


# ==========================================
# 保存模型
# ==========================================

def save_all_models(trained_models, scaler, feature_names, optimized_params=None, save_dir='saved_models'):
    """保存所有训练好的模型和预处理器"""
    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # 保存所有模型
    for name, model in trained_models.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        pkl_path = os.path.join(save_dir, f"{safe_name}.pkl")

        try:
            joblib.dump(model, pkl_path)
            print(f"已保存模型: {pkl_path}")

            # XGBoost原生格式
            if hasattr(model, 'save_model'):
                native_path = os.path.join(save_dir, f"{safe_name}_xgb.model")
                model.save_model(native_path)
                print(f"已保存XGBoost原生格式: {native_path}")

            # LightGBM原生格式
            elif hasattr(model, 'booster_') and hasattr(model.booster_, 'save_model'):
                native_path = os.path.join(save_dir, f"{safe_name}_lgb.txt")
                model.booster_.save_model(native_path)
                print(f"已保存LightGBM原生格式: {native_path}")

        except Exception as e:
            print(f"保存模型 {name} 时出错: {e}")

    # 保存scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"已保存标准化器: {scaler_path}")

    # 保存特征名称
    feature_path = os.path.join(save_dir, 'feature_names.pkl')
    joblib.dump(feature_names, feature_path)
    print(f"已保存特征名称: {feature_path}")

    # 保存优化参数
    if optimized_params:
        params_path = os.path.join(save_dir, 'optimized_params.pkl')
        joblib.dump(optimized_params, params_path)
        print(f"已保存优化参数: {params_path}")


def generate_results_table(results, save_path='train_results.csv'):
    """生成并保存结果表"""
    print("\n" + "=" * 60)
    print("生成结果表")
    print("=" * 60)

    df = pd.DataFrame(results).T
    key_metrics = ['AUC', 'Accuracy', 'F1-Score', 'Sensitivity', 'Specificity',
                   'Precision', 'Recall', 'PPV', 'NPV', 'MCC', 'Kappa', 'Brier Score']

    cols = [col for col in key_metrics if col in df.columns] + \
           [col for col in df.columns if col not in key_metrics]

    df = df[cols].round(4).sort_values('AUC', ascending=False)

    print(df.to_string())
    df.to_csv(save_path)
    print(f"\n结果已保存: {save_path}")

    return df


# ==========================================
# 执行训练流程
# ==========================================

print("\n" + "=" * 60)
print("执行训练流程: 基础模型 + Optuna优化 + 集成学习")
print("=" * 60)
###################################

# 补充缺失的重要 import (防止未定义错误)
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier

# 1. 设置数据路径 (请确认文件名正确)
data_file_path = 'train_data.csv'

# 2. 执行数据准备
X_train, X_test, y_train, y_test, feature_names = prepare_data(data_file_path)

# 3. 执行类别不平衡处理 (定义 X_train_resampled)
X_train_resampled, y_train_resampled = handle_imbalance(X_train, y_train, method='smote')
# 更新主变量
X_train = X_train_resampled
y_train = y_train_resampled

# 4. 执行标准化 (定义 X_train_scaled, 这一步解决了你之前 "未定义" 的报错)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# ===================================================
# Step 2.5: 特征选择 (Feature Selection) - 过滤 + RFE
# ===================================================
print("\n" + "=" * 50)
print("Step 2.5: 执行特征选择流程")
print("=" * 50)

# 备份一份数据用于筛选计算
X_train_selection_temp = X_train_scaled.copy()
X_test_selection_temp = X_test_scaled.copy()
current_feat_names = list(feature_names)  # 此时是完整的特征列表

# --- A. 方差过滤 ---
selector_var = VarianceThreshold(threshold=0)
X_train_selection_temp = selector_var.fit_transform(X_train_selection_temp)
mask_var = selector_var.get_support()
# 更新
X_test_selection_temp = X_test_selection_temp[:, mask_var]
selected_feat_names = [f for f, k in zip(current_feat_names, mask_var) if k]
print(f"1. 方差过滤后特征数: {len(current_feat_names)} -> {len(selected_feat_names)}")

# --- B. 相关性过滤 (>0.9) ---
df_corr = pd.DataFrame(X_train_selection_temp, columns=selected_feat_names)
corr_matrix = df_corr.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.90)]

if to_drop:
    print(f"   -> 移除高共线性特征: {to_drop}")
    keep_idx = [i for i, f in enumerate(selected_feat_names) if f not in to_drop]
    X_train_selection_temp = X_train_selection_temp[:, keep_idx]
    X_test_selection_temp = X_test_selection_temp[:, keep_idx]
    selected_feat_names = [selected_feat_names[i] for i in keep_idx]
print(f"2. 共线性处理后特征数: {len(selected_feat_names)}")

# --- C. RFE 保留 Top 20 ---
target_n = 20
if len(selected_feat_names) > target_n:
    print(f"3. 执行 RFE，筛选 Top {target_n} 特征...")
    rf_rfe = RandomForestClassifier(n_jobs=1, random_state=42, n_estimators=50)
    rfe = RFE(estimator=rf_rfe, n_features_to_select=target_n, step=1)
    rfe.fit(X_train_selection_temp, y_train)
    mask_rfe = rfe.support_

    selected_feat_names = [f for f, k in zip(selected_feat_names, mask_rfe) if k]
    print("   -> RFE完成")

print(f"✅ 最终特征列表: {selected_feat_names}")


print("\n🔄 [Fix] 正在基于筛选后的特征重新拟合 Scaler...")

# 1. 找回未标准化的原始数据 (X_train)


if isinstance(X_train, np.ndarray):
    # 如果是 numpy 数组，先转 DataFrame 以便按列名筛选
    df_train_unscaled = pd.DataFrame(X_train, columns=feature_names)
    df_test_unscaled = pd.DataFrame(X_test, columns=feature_names)
else:
    # 如果已经是 DataFrame
    df_train_unscaled = X_train.copy()
    df_test_unscaled = X_test.copy()

# 2. 只保留筛选出的特征 (Unscaled)
X_train_final_unscaled = df_train_unscaled[selected_feat_names]
X_test_final_unscaled = df_test_unscaled[selected_feat_names]

# 3. 创建全新的 Scaler 并拟合
new_scaler = StandardScaler()
X_train_scaled = new_scaler.fit_transform(X_train_final_unscaled)  # 覆盖主变量
X_test_scaled = new_scaler.transform(X_test_final_unscaled)  # 覆盖主变量

# 4. 更新全局变量
scaler = new_scaler
feature_names = selected_feat_names

print(f"   Scaler 已更新，内部特征数 (n_features_in_): {scaler.n_features_in_}")
print("   数据形状已更新:", X_train_scaled.shape)

# ==========================================
# 保存筛选后的特征名与新的标准化器
# ==========================================
# 注意：这里保存为 feature_names1.pkl 和 scaler1.pkl 以匹配您的验证代码

os.makedirs('saved_models', exist_ok=True)
joblib.dump(feature_names, os.path.join('saved_models', 'feature_names1.pkl'))
joblib.dump(scaler, os.path.join('saved_models', 'scaler1.pkl'))
print("✅ 新的 Scaler 和特征名已保存到 saved_models/scaler1.pkl")
# ==========================================
# 后续：Optuna 优化与模型训练
# ==========================================

# 5. Optuna 优化 (使用更新后的 X_train_scaled)
best_xgb, xgb_params = optimize_xgboost(X_train_scaled, y_train)
best_lgb, lgb_params = optimize_lightgbm(X_train_scaled, y_train)
best_rf, rf_params = optimize_random_forest(X_train_scaled, y_train)

optimized_models = {
    'XGBoost Optimized': best_xgb,
    'LightGBM Optimized': best_lgb,
    'Random Forest Optimized': best_rf
}

# 6. 定义基础模型
base_models = define_models()

# 7. 创建集成模型
ensemble_models = create_ensemble_models(base_models, optimized_models)

# 8. 合并所有模型
all_models = {**base_models, **optimized_models, **ensemble_models}

# 9. 训练并评估
results, trained_models, predictions = train_and_evaluate_models(
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, all_models
)

# 10. 可视化
plot_roc_curves(trained_models, X_test_scaled, y_test)
plot_metrics_heatmap(results)
plot_model_comparison_bar(results)

# 11. 生成表格
df_results = generate_results_table(results)

# 12. 保存所有模型 (这一步也会保存 feature_names.pkl 和 scaler.pkl 作为备份)
save_all_models(trained_models, scaler, feature_names,
                optimized_params={**xgb_params, **lgb_params, **rf_params})

print("\n🎉 所有流程执行完毕！")

# ==========================================
# 期刊级别可视化 - 独立图表版本
# ==========================================
# ==========================================
# 期刊级别可视化 - 统一按测试集AUC排序 + 多格式保存 + CSV导出
# ==========================================
# ==========================================
# 期刊级别可视化 - 终极优化版
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# ========== Lancet/NEJM 风格配色 ==========
PALETTE = {
    "main": ["#0072B5", "#BC3C29", "#20854E", "#E18727",
             "#6F99AD", "#925E9F", "#FFDC91", "#7876B1"],
    "accent": "#0072B5",
    "danger": "#BC3C29",
    "success": "#20854E",
    "bg": "#FAFAFA",
    "grid": "#CCCCCC",
    "text": "#333333",
}

# ========== 基础工具 ==========
def mj_set_style():
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "microsoft yahei",
        "axes.unicode_minus": False,
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "white",
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#555555",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "grid.alpha": 0.3,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "lines.antialiased": True,
    })

def mj_despine(ax, top=True, right=True):
    """移除上/右边框（期刊常用风格）"""
    ax.spines["top"].set_visible(not top)
    ax.spines["right"].set_visible(not right)

def mj_ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def mj_save_fig(fig, out_dir, base_name, formats=("png", "pdf", "svg"), dpi=600):
    mj_ensure_dir(out_dir)
    saved = []
    for fmt in formats:
        p = os.path.join(out_dir, f"{base_name}.{fmt}")
        fig.savefig(p, bbox_inches="tight",
                    dpi=(dpi if fmt in ("png", "tif", "tiff", "jpg") else None),
                    facecolor="white", edgecolor="none")
        saved.append(fmt)
    plt.close(fig)
    print(f"  ✅ {base_name} -> {', '.join(saved)}")

def mj_to_csv(df, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  📄 {os.path.basename(out_path)}")

def mj_has_proba(model, X):
    if not hasattr(model, "predict_proba"):
        return False
    try:
        p = model.predict_proba(X[:1])
        return p is not None and p.ndim == 2 and p.shape[1] >= 2
    except Exception:
        return False

def mj_rank(results, trained_models, X_test=None, require_proba=False):
    df = pd.DataFrame(results).T
    df = df.loc[df.index.intersection(trained_models.keys())].copy()
    if require_proba:
        keep = [m for m in df.index if mj_has_proba(trained_models[m], X_test)]
        df = df.loc[keep]
    df = df.sort_values("AUC", ascending=False)
    return df.index.tolist(), df

def mj_get_palette(n):
    base = PALETTE["main"]
    if n <= len(base):
        return base[:n]
    return sns.color_palette("husl", n_colors=n)

# ========================================================
# 01 DCA（双面板：曲线 + 热力图）
# ========================================================
def mj_net_benefit(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    if thr >= 1.0:
        return 0.0
    return (tp / n) - (fp / n) * (thr / (1 - thr))

def mj_plot_01_dca(trained_models, X_test, y_test, results, out_dir,
                   top_n=5, formats=("png", "pdf", "svg")):
    print("\n📊 01 Decision Curve Analysis")
    mj_set_style()
    ranked, df_auc = mj_rank(results, trained_models, X_test=X_test, require_proba=True)
    top = ranked[:top_n]
    thresholds = np.arange(0.01, 0.99, 0.01)
    colors = mj_get_palette(len(top))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    csv_rows, csv_key = [], []
    key_thr = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_nb = {}

    for name, color in zip(top, colors):
        yp = trained_models[name].predict_proba(X_test)[:, 1]
        nb = [mj_net_benefit(y_test, yp, t) for t in thresholds]
        all_nb[name] = nb
        ax1.plot(thresholds, nb, color=color, lw=2.5,
                 label=f"{name} (AUC={df_auc.loc[name,'AUC']:.3f})")

        for t, n_b in zip(thresholds, nb):
            csv_rows.append({"Model": name, "AUC": df_auc.loc[name, "AUC"],
                             "Threshold": float(t), "Net_Benefit": float(n_b)})
        rec = {"Model": name, "AUC": df_auc.loc[name, "AUC"]}
        for kt in key_thr:
            rec[f"NB@{kt:.1f}"] = float(mj_net_benefit(y_test, yp, kt))
        csv_key.append(rec)

    prev = float(np.mean(y_test))
    treat_all = [prev - (1 - prev) * (t / (1 - t)) for t in thresholds]
    ax1.plot(thresholds, treat_all, "k--", lw=1.8, label="Treat All")
    ax1.axhline(0, color="gray", ls=":", lw=1.8, label="Treat None")

    # 有用区域着色
    ax1.axvspan(0.1, 0.4, alpha=0.06, color=PALETTE["accent"], label="Clinical range")

    ax1.set_xlim(0, 0.8)
    # 固定Y轴范围与刻度：-0.05 ~ 0.30，步长0.05
    ax1.set_ylim(-0.05, 0.30)
    ax1.set_yticks(np.arange(-0.05, 0.301, 0.05))
    ax1.set_xlabel("Threshold Probability")
    ax1.set_ylabel("Net Benefit")
    ax1.set_title("Decision Curve Analysis")
    ax1.legend(loc="upper right", frameon=True, fontsize=8)
    ax1.grid(True, alpha=0.2)
    mj_despine(ax1)

    # 热力图
    hm_data = []
    for name in top:
        for kt in key_thr:
            idx = int(kt * 100) - 1
            hm_data.append({"Model": name, "Threshold": kt, "NB": all_nb[name][idx]})
    hm_df = pd.DataFrame(hm_data).pivot(index="Model", columns="Threshold", values="NB")

    sns.heatmap(hm_df, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
                linewidths=1.5, linecolor="white", ax=ax2,
                cbar_kws={"label": "Net Benefit", "shrink": 0.8},
                annot_kws={"size": 10, "weight": "bold"})
    ax2.set_title("Net Benefit at Key Thresholds")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("")

    fig.suptitle("Decision Curve Analysis (Top Models by Test AUC)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    mj_save_fig(fig, out_dir, "01_decision_curve_analysis", formats, 600)
    mj_to_csv(pd.DataFrame(csv_rows), os.path.join(out_dir, "01_dca_curve_points.csv"))
    mj_to_csv(pd.DataFrame(csv_key), os.path.join(out_dir, "01_dca_key_thresholds.csv"))

# ========================================================
# 02 Calibration + H-L + 概率分布直方图
# ========================================================
def mj_hosmer_lemeshow(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bi = np.clip(np.digitize(y_prob, bins[:-1]) - 1, 0, n_bins - 1)
    yt, yp = np.asarray(y_true), np.asarray(y_prob)
    obs, exp, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for i in range(n_bins):
        m = bi == i
        cnt[i] = m.sum()
        if cnt[i] > 0:
            obs[i] = (yt[m] == 1).sum()
            exp[i] = yp[m].sum()
    m = cnt > 0
    chi2 = np.sum((obs[m] - exp[m]) ** 2 / (exp[m] * (1 - exp[m] / cnt[m]) + 1e-10))
    return float(chi2), float(1 - stats.chi2.cdf(chi2, n_bins - 2))

def mj_plot_02_calibration(trained_models, X_test, y_test, results, out_dir,
                           top_n=6, formats=("png", "pdf", "svg")):
    print("\n📊 02 Calibration + Hosmer-Lemeshow")
    mj_set_style()
    ranked, df_auc = mj_rank(results, trained_models, X_test=X_test, require_proba=True)
    top = ranked[:top_n]

    nc = 3
    nr = int(np.ceil(len(top) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(18, 6.5 * nr))
    axes = np.array(axes).reshape(-1)
    colors = mj_get_palette(len(top))

    csv_rows = []
    for i, (name, color) in enumerate(zip(top, colors)):
        ax = axes[i]
        yp = trained_models[name].predict_proba(X_test)[:, 1]
        fp, mp = calibration_curve(y_test, yp, n_bins=10, strategy="uniform")
        chi2, p = mj_hosmer_lemeshow(y_test, yp)
        brier = brier_score_loss(y_test, yp)

        # 校准曲线
        ax.plot(mp, fp, "o-", color=color, lw=2.5, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)

        # 置信带（简化：基于二项分布标准误）
        for x_pt, y_pt in zip(mp, fp):
            n_in_bin = max(1, int(len(yp) / 10))
            se = np.sqrt(y_pt * (1 - y_pt) / n_in_bin)
            ax.fill_between([x_pt - 0.02, x_pt + 0.02],
                            max(0, y_pt - 1.96 * se), min(1, y_pt + 1.96 * se),
                            alpha=0.15, color=color)

        # 底部概率分布直方图
        ax2 = ax.twinx()
        ax2.hist(yp[np.asarray(y_test) == 0], bins=30, range=(0, 1),
                 alpha=0.25, color=PALETTE["main"][0], label="Neg")
        ax2.hist(yp[np.asarray(y_test) == 1], bins=30, range=(0, 1),
                 alpha=0.25, color=PALETTE["main"][1], label="Pos")
        ax2.set_ylim(0, ax2.get_ylim()[1] * 5)
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)

        # 统计信息
        hl_sig = "✓ Good" if p > 0.05 else "✗ Poor"
        ax.text(0.04, 0.96,
                f"AUC = {df_auc.loc[name,'AUC']:.3f}\n"
                f"H-L χ² = {chi2:.2f}\n"
                f"p = {p:.4f} ({hl_sig})\n"
                f"Brier = {brier:.4f}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.92, ec="#CCCCCC"))

        ax.set_title(f"#{i+1} {name}", fontsize=11)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed proportion")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        mj_despine(ax)

        csv_rows.append({"Model": name, "AUC": df_auc.loc[name, "AUC"],
                         "HL_Chi2": chi2, "HL_p": p, "HL_Good": p > 0.05, "Brier": brier})

    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Calibration Curves with Hosmer–Lemeshow Test (Top by Test AUC)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    mj_save_fig(fig, out_dir, "02_calibration_curves_hl", formats, 600)
    mj_to_csv(pd.DataFrame(csv_rows), os.path.join(out_dir, "02_calibration_results.csv"))

# ========================================================
# 03 Forest Plot（经典菱形 + 右侧数值列）
# ========================================================
def mj_bootstrap_ci(y_true, y_prob, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_prob)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[idx])) < 2:
            continue
        aucs.append(roc_auc_score(yt[idx], yp[idx]))
    if len(aucs) < 50:
        return np.nan, np.nan
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))

def mj_plot_03_forest(trained_models, X_test, y_test, results, out_dir,
                      n_bootstrap=2000, formats=("png", "pdf", "svg")):
    print("\n📊 03 Forest Plot (Bootstrap 95% CI)")
    mj_set_style()
    ranked, df_auc = mj_rank(results, trained_models, X_test=X_test, require_proba=True)

    rows = []
    for name in ranked:
        yp = trained_models[name].predict_proba(X_test)[:, 1]
        auc = float(df_auc.loc[name, "AUC"])
        lo, hi = mj_bootstrap_ci(y_test, yp, n=n_bootstrap)
        rows.append([name, auc, lo, hi])

    dfp = pd.DataFrame(rows, columns=["Model", "AUC", "CI_low", "CI_high"])
    dfp = dfp.sort_values("AUC", ascending=False).reset_index(drop=True)

    n = len(dfp)
    fig, ax = plt.subplots(figsize=(13, max(7, 0.48 * n + 2)))
    y = np.arange(n)

    for i, r in dfp.iterrows():
        auc, lo, hi = r["AUC"], r["CI_low"], r["CI_high"]
        # CI 线
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color="#555555", lw=1.8, zorder=1)
            ax.plot([lo, lo], [i - 0.12, i + 0.12], color="#555555", lw=1.2)
            ax.plot([hi, hi], [i - 0.12, i + 0.12], color="#555555", lw=1.2)

        # 菱形
        diamond_size = 0.15
        diamond = plt.Polygon([
            [auc, i + diamond_size], [auc + 0.008, i],
            [auc, i - diamond_size], [auc - 0.008, i]
        ], color=PALETTE["accent"], ec="black", lw=0.8, zorder=3)
        ax.add_patch(diamond)

        # 右侧数值
        ci_text = f"{lo:.3f}–{hi:.3f}" if np.isfinite(lo) else "N/A"
        ax.text(1.04, i, f"{auc:.3f}  ({ci_text})",
                va="center", fontsize=9, fontfamily="monospace",
                transform=ax.get_yaxis_transform())

    ax.axvline(0.5, color=PALETTE["danger"], ls="--", lw=1.2, alpha=0.5)
    ax.axvline(0.8, color=PALETTE["success"], ls=":", lw=1, alpha=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels(dfp["Model"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Test AUC")
    ax.set_title("Forest Plot — AUC with Bootstrap 95% CI")
    ax.set_xlim(0.4, 1.01)
    ax.grid(True, axis="x", alpha=0.2)
    mj_despine(ax)

    # 右侧表头
    ax.text(1.04, -0.7, "AUC (95% CI)", va="center", fontsize=9,
            fontweight="bold", fontfamily="monospace",
            transform=ax.get_yaxis_transform())

    mj_save_fig(fig, out_dir, "03_forest_plot", formats, 600)
    mj_to_csv(dfp, os.path.join(out_dir, "03_forest_plot_ci.csv"))

# ========================================================
# 04 Feature Importance（Top4 + 渐变色 + 累积贡献）
# ========================================================
def mj_plot_04_feature_importance(trained_models, feature_names, results, out_dir,
                                  X_ref, y_ref, top_n_features=15, perm_repeats=20,
                                  formats=("png", "pdf", "svg")):
    print("\n📊 04 Feature Importance (Top4 by Test AUC)")
    mj_set_style()
    ranked, df_auc = mj_rank(results, trained_models, require_proba=False)
    top4 = ranked[:4]

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = axes.flatten()

    csv_rows = []
    for i in range(4):
        ax = axes[i]
        if i >= len(top4):
            ax.set_visible(False)
            continue

        name = top4[i]
        model = trained_models[name]

        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            method = "Built-in"
        elif hasattr(model, "coef_"):
            imp = np.abs(np.asarray(model.coef_[0], dtype=float))
            method = "Coefficient"
        else:
            r = permutation_importance(model, X_ref, y_ref, scoring="roc_auc",
                                       n_repeats=perm_repeats, random_state=42, n_jobs=-1)
            imp = np.abs(np.asarray(r.importances_mean, dtype=float))
            method = "Permutation"

        if np.allclose(imp.sum(), 0):
            ax.text(0.5, 0.5, "All importances = 0", ha="center", va="center")
            ax.set_title(f"#{i+1} {name}")
            continue

        imp = imp / (imp.sum() + 1e-12)
        idx = np.argsort(imp)[::-1][:top_n_features]
        feats = [feature_names[j] for j in idx][::-1]
        vals = imp[idx][::-1]

        # 渐变色
        cmap = plt.cm.Blues
        norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        bar_colors = [cmap(0.3 + 0.6 * v) for v in norm_vals]

        bars = ax.barh(range(len(feats)), vals, color=bar_colors,
                       edgecolor="white", lw=0.8, alpha=0.92)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=9)
        ax.set_xlabel("Normalized importance")
        ax.set_title(f"#{i+1}  {name}\nAUC={df_auc.loc[name,'AUC']:.3f} | {method}",
                     fontsize=11)
        ax.grid(True, axis="x", alpha=0.2)
        mj_despine(ax)

        # 百分比标注
        for bar, vv in zip(bars, vals):
            pct = vv * 100
            ax.text(vv + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8.5, color="#333333")

        # 累积贡献线（右侧次坐标）
        sorted_vals = imp[idx]  # 降序
        cumsum = np.cumsum(sorted_vals)
        ax2 = ax.twiny()
        ax2.plot(cumsum[::-1], range(len(feats)), "o-",
                 color=PALETTE["danger"], lw=1.5, markersize=4, alpha=0.7)
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Cumulative contribution", fontsize=9, color=PALETTE["danger"])
        ax2.tick_params(axis="x", colors=PALETTE["danger"], labelsize=8)
        ax2.spines["top"].set_color(PALETTE["danger"])
        ax2.spines["top"].set_alpha(0.5)

        for rank, j in enumerate(idx, 1):
            csv_rows.append({"Model": name, "AUC": df_auc.loc[name, "AUC"],
                             "Rank": rank, "Feature": feature_names[j],
                             "Importance": float(imp[j]),
                             "Cumulative": float(np.sum(imp[idx[:rank]])),
                             "Method": method})

    fig.suptitle(f"Feature Importance — Top4 Models by Test AUC (Top {top_n_features} features)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    mj_save_fig(fig, out_dir, "04_feature_importance_top4", formats, 600)
    mj_to_csv(pd.DataFrame(csv_rows), os.path.join(out_dir, "04_feature_importance_top4.csv"))

# ========================================================
# 05 Sensitivity vs Specificity（Youden 等值线 + F1 映射）
# ========================================================
def mj_plot_05_sens_spec(results, out_dir, formats=("png", "pdf", "svg")):
    print("\n📊 05 Sensitivity vs Specificity")
    mj_set_style()
    df = pd.DataFrame(results).T.sort_values("AUC", ascending=False)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))

    # Youden 等值线
    for youden_val in [0.2, 0.4, 0.6, 0.8]:
        x_line = np.linspace(0.5, 1, 100)
        y_line = youden_val - x_line + 1
        valid = (y_line >= 0.5) & (y_line <= 1.05)
        ax.plot(x_line[valid], y_line[valid], ":", color="#AAAAAA", lw=0.8, alpha=0.6)
        # 标注
        mid = len(x_line[valid]) // 2
        if np.any(valid):
            ax.text(x_line[valid][mid], y_line[valid][mid] + 0.015,
                    f"J={youden_val:.1f}", fontsize=7, color="#999999", rotation=-45)

    colors = mj_get_palette(len(df))
    for (model, row), c in zip(df.iterrows(), colors):
        f1 = row.get("F1-Score", row["AUC"])
        size = max(80, f1 * 600)
        ax.scatter(row["Specificity"], row["Sensitivity"],
                   s=size, color=c, alpha=0.8,
                   edgecolors="black", linewidth=0.6, zorder=3,
                   label=f"{model} (AUC={row['AUC']:.3f})")

    ax.scatter([1], [1], s=320, c="red", marker="*",
               edgecolors="black", linewidths=1.0, label="Ideal", zorder=10)
    ax.set_xlim(0.45, 1.04); ax.set_ylim(0.45, 1.04)
    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Sensitivity vs Specificity\n(bubble size ∝ F1-Score, dashed lines = Youden index)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0), fontsize=8, frameon=True)
    mj_despine(ax)

    mj_save_fig(fig, out_dir, "05_sensitivity_specificity", formats, 600)

    cols = [c for c in ["AUC", "Sensitivity", "Specificity", "PPV", "NPV",
                        "Accuracy", "F1-Score", "Youden Index"] if c in df.columns]
    csv_df = df[cols].copy()
    csv_df.insert(0, "Model", csv_df.index)
    csv_df = csv_df.reset_index(drop=True)
    mj_to_csv(csv_df, os.path.join(out_dir, "05_sens_spec_data.csv"))

# ========================================================
# 06 PR Curves（iso-F1 + 填充 + 面积标注）
# ========================================================
def mj_plot_06_pr(trained_models, X_test, y_test, results, out_dir,
                  top_n=10, formats=("png", "pdf", "svg")):
    print("\n📊 06 Precision-Recall Curves")
    mj_set_style()
    ranked, df_auc = mj_rank(results, trained_models, X_test=X_test, require_proba=True)
    top = ranked[:top_n]

    fig, ax = plt.subplots(figsize=(10.5, 8))
    colors = mj_get_palette(len(top))

    # iso-F1 曲线
    for f1_val in [0.3, 0.5, 0.7, 0.9]:
        r_vals = np.linspace(0.01, 1, 200)
        p_vals = f1_val * r_vals / (2 * r_vals - f1_val)
        valid = (p_vals > 0) & (p_vals <= 1)
        ax.plot(r_vals[valid], p_vals[valid], ":", color="#BBBBBB", lw=0.7, alpha=0.5)
        idx_mid = np.argmin(np.abs(r_vals[valid] - 0.95))
        if np.any(valid):
            ax.text(r_vals[valid][idx_mid], p_vals[valid][idx_mid] + 0.02,
                    f"F1={f1_val}", fontsize=7, color="#999999")

    csv_rows = []
    for name, color in zip(top, colors):
        yp = trained_models[name].predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, yp)
        ap = average_precision_score(y_test, yp)

        ax.plot(rec, prec, color=color, lw=2.2,
                label=f"{name} (AUC={df_auc.loc[name,'AUC']:.3f}, AP={ap:.3f})")
        ax.fill_between(rec, prec, alpha=0.05, color=color)

        for p, r in zip(prec, rec):
            csv_rows.append({"Model": name, "AUC": df_auc.loc[name, "AUC"],
                             "AP": ap, "Recall": float(r), "Precision": float(p)})

    bl = float(np.mean(y_test))
    ax.axhline(bl, color="black", ls="--", lw=1.2, alpha=0.5,
               label=f"Baseline (prev={bl:.3f})")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves (Top by Test AUC)\nDotted lines = iso-F1 contours")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower left", fontsize=8, ncol=2, frameon=True)
    mj_despine(ax)

    mj_save_fig(fig, out_dir, "06_precision_recall_curves", formats, 600)
    mj_to_csv(pd.DataFrame(csv_rows), os.path.join(out_dir, "06_pr_curve_points.csv"))

# ========================================================
# 07 Confusion Matrix（计数 + 百分比 双标注）
# ========================================================
def mj_plot_07_cm(trained_models, X_test, y_test, results, out_dir,
                  formats=("png", "pdf", "svg")):
    print("\n📊 07 Confusion Matrix (Best by Test AUC)")
    mj_set_style()
    ranked, df_auc = mj_rank(results, trained_models, require_proba=False)
    best = ranked[0]
    model = trained_models[best]

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    sen = tp / (tp + fn) if (tp + fn) else 0
    spe = tn / (tn + fp) if (tn + fp) else 0
    ppv_val = tp / (tp + fp) if (tp + fp) else 0
    npv_val = tn / (tn + fn) if (tn + fn) else 0
    acc = (tp + tn) / total

    # 双标注文本
    cm_pct = cm / total * 100
    annot = np.array([
        [f"{cm[0,0]}\n({cm_pct[0,0]:.1f}%)", f"{cm[0,1]}\n({cm_pct[0,1]:.1f}%)"],
        [f"{cm[1,0]}\n({cm_pct[1,0]:.1f}%)", f"{cm[1,1]}\n({cm_pct[1,1]:.1f}%)"]
    ])

    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Predicted\nNegative", "Predicted\nPositive"],
                yticklabels=["Actual\nNegative", "Actual\nPositive"],
                linewidths=2, linecolor="white", ax=ax, cbar=False,
                annot_kws={"size": 14, "weight": "bold"})

    ax.set_title(f"Best Model: {best}\nTest AUC = {df_auc.loc[best,'AUC']:.3f}",
                 fontsize=13, pad=15)

    # 右侧指标卡片
    metrics_text = (
        f"━━━ Performance ━━━\n\n"
        f"  Accuracy   {acc:.3f}\n"
        f"  Sensitivity {sen:.3f}\n"
        f"  Specificity {spe:.3f}\n"
        f"  PPV        {ppv_val:.3f}\n"
        f"  NPV        {npv_val:.3f}\n\n"
        f"━━━ Counts ━━━━━\n\n"
        f"  TP={tp}  FP={fp}\n"
        f"  FN={fn}  TN={tn}\n"
        f"  Total={total}"
    )
    ax.text(1.08, 0.5, metrics_text,
            transform=ax.transAxes, fontsize=10, va="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="#F7F7F7", ec="#CCCCCC", lw=0.8))

    mj_save_fig(fig, out_dir, "07_confusion_matrix_best", formats, 600)
    mj_to_csv(pd.DataFrame([{
        "Model": best, "AUC": df_auc.loc[best, "AUC"],
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Accuracy": acc, "Sensitivity": sen, "Specificity": spe,
        "PPV": ppv_val, "NPV": npv_val
    }]), os.path.join(out_dir, "07_confusion_matrix_best.csv"))
    print(f"  🏆 最优: {best} | AUC={df_auc.loc[best,'AUC']:.4f}")

# ========================================================
# 00 总排名 CSV
# ========================================================
def mj_export_00(results, out_dir):
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    df = df.sort_values("AUC", ascending=False)
    mj_to_csv(df, os.path.join(out_dir, "00_model_ranking_by_test_auc.csv"))

# ========================================================
# 主入口
# ========================================================
def generate_medical_figures(
    trained_models, X_test, y_test, results, feature_names,
    output_dir="medical_figures",
    dca_top=5, calib_top=6, pr_top=10,
    forest_bootstrap=2000,
    feat_top_n=15, perm_repeats=20,
    formats=("png", "pdf", "svg")
):
    mj_ensure_dir(output_dir)
    print("\n" + "=" * 70)
    print("🎨 开始生成医学期刊级别可视化（终极优化版）")
    print("=" * 70)

    mj_export_00(results, output_dir)
    mj_plot_01_dca(trained_models, X_test, y_test, results, output_dir, top_n=dca_top, formats=formats)
    mj_plot_02_calibration(trained_models, X_test, y_test, results, output_dir, top_n=calib_top, formats=formats)
    mj_plot_03_forest(trained_models, X_test, y_test, results, output_dir, n_bootstrap=forest_bootstrap, formats=formats)
    mj_plot_04_feature_importance(trained_models, feature_names, results, output_dir,
                                  X_ref=X_test, y_ref=y_test,
                                  top_n_features=feat_top_n, perm_repeats=perm_repeats, formats=formats)
    mj_plot_05_sens_spec(results, output_dir, formats=formats)
    mj_plot_06_pr(trained_models, X_test, y_test, results, output_dir, top_n=pr_top, formats=formats)
    mj_plot_07_cm(trained_models, X_test, y_test, results, output_dir, formats=formats)

    print("\n" + "=" * 70)
    print("✅ 医学期刊级别可视化全部完成！")
    print("=" * 70)
    print(f"📁 输出: {output_dir}/  |  格式: {formats}")
    print("📋 00-07 图表 + 配套 CSV 已全部生成")

# ========== 调用 ==========
generate_medical_figures(
    trained_models=trained_models,
    X_test=X_test_scaled,
    y_test=y_test,
    results=results,
    feature_names=feature_names,
    output_dir="medical_figures",
    dca_top=5,
    calib_top=6,
    pr_top=10,
    forest_bootstrap=2000,
    feat_top_n=15,
    perm_repeats=20,
    formats=("png", "pdf", "svg")
)
