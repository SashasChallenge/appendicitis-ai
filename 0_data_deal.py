
import matplotlib
matplotlib.use('Agg')    # ← 加这行
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 读取数据
file_path = 'clean data all.xlsx'
try:
    df = pd.read_excel(file_path)
    print("数据读取成功！")
except FileNotFoundError:
    print(f"找不到文件: {file_path}")
    exit()

# ==========================================
# 步骤 1: 特征筛选 (删除不该保留的特征)
# ==========================================
# 基于之前的分析，这些列包含未来信息(泄露)或无关信息，必须删除
cols_to_drop = [
    'Age1',
]


existing_drop_cols = [col for col in cols_to_drop if col in df.columns]
df.drop(columns=existing_drop_cols, inplace=True)
print(f"\n已删除 {len(existing_drop_cols)} 个无效/泄露特征列。")

# ==========================================
# 步骤 2: 处理由 'na' 导致的类型错误
# ==========================================

# 定义离散变量（分类变量），这些通常用众数填充，或者保持原样
#x19到x38是离散变量
discrete_cols = [
                 'Gender','Diarrhea', 'Hypertension', 'Atrial Fibrillation', 'Coronary Heart Diseases', 'Peripheral_vascular_disease', 'Cerebrovascular_disease',
                 'Chronic_pulmonary_disease', 'Peptic_ulcer_disease', 'Diabetes', 'Renal_disease', 'Severe_liver_disease', 'Charlson_comorbidity_index',
                ]

# 确保列表中的列都在df中
discrete_cols = [c for c in discrete_cols if c in df.columns]

# 定义连续变量（数值变量）：剩下的就是连续变量
continuous_cols = [col for col in df.columns if col not in discrete_cols]

print("\n正在处理连续变量中的 'na' 字符...")

for col in continuous_cols:
    # 核心修复代码：errors='coerce' 会将 'na'、'NA' 或其他无法转数字的文本强制变成 NaN (空值)
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # 计算中位数填充
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"  - {col}: 类型已修正，空值已用中位数 {median_val} 填充")

# ==========================================
# 其他离散变量如果有空值，通常用众数(mode)填充
for col in discrete_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

# ==========================================
# 步骤 4: 最终检查
# ==========================================
print("\n数据预处理完成！")
print(df.info())
print("\n最终数据预览:")
print(df.head())

# 如果需要保存清洗后的数据
df.to_excel("A_B_cleaned_data_for_model_2026_1_14.xlsx", index=False)


# 2. 设置参数
target_variable = 'Target'  # 目标变量名称
random_seed = 42            # 设置随机种子以保证结果可复现
split_ratio = 0.2           # 验证集比例（此处设为20%作为外部验证，可根据需要调整）

# 检查目标变量是否存在
if target_variable not in df.columns:
    print(f"错误：数据中未找到目标变量列 '{target_variable}'")
else:
    # 3. 划分数据集
    # 使用 stratify 参数可以保证训练集和验证集中目标变量的分布一致（适用于分类任务）
    # 如果是回归任务，请去掉 stratify=df[target_variable]
    train_df, val_df = train_test_split(
        df,
        test_size=split_ratio,
        random_state=random_seed,
        stratify=df[target_variable]
    )

    # 4. 定义文件名
    train_filename = 'train_data.csv'
    val_filename = 'external_validation_data.csv'

    # 5. 保存文件
    train_df.to_csv(train_filename, index=False)
    val_df.to_csv(val_filename, index=False)

    print("-" * 30)
    print(f"处理完成！")
    print(f"随机种子已设置为: {random_seed}")
    print(f"训练集已保存为: {train_filename} (行数: {len(train_df)})")
    print(f"外部验证集已保存为: {val_filename} (行数: {len(val_df)})")

from scipy import stats

# ==========================================
# 步骤 5: 显著性组间对比分析
# ==========================================
print("\n" + "=" * 60)
print("显著性组间对比分析")
print("=" * 60)

# 获取目标变量的唯一值
groups = df[target_variable].unique()
print(f"\n目标变量 '{target_variable}' 的分组: {groups}")

# 初始化结果存储
comparison_results = []

# 对连续变量进行组间比较
print("\n【连续变量组间对比】")
print("-" * 60)

for col in continuous_cols:
    if col == target_variable:  # 跳过目标变量本身
        continue

    # 按目标变量分组
    group_data = [df[df[target_variable] == g][col].dropna() for g in groups]

    # 根据分组数量选择检验方法
    if len(groups) == 2:
        # 两组比较：t检验和Mann-Whitney U检验
        t_stat, t_pvalue = stats.ttest_ind(group_data[0], group_data[1])
        u_stat, u_pvalue = stats.mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')

        means = [g.mean() for g in group_data]
        stds = [g.std() for g in group_data]

        result = {
            '变量': col,
            f'组{groups[0]}_均值±标准差': f'{means[0]:.4f}±{stds[0]:.4f}',
            f'组{groups[1]}_均值±标准差': f'{means[1]:.4f}±{stds[1]:.4f}',
            't检验_p值': f'{t_pvalue:.4f}',
            'U检验_p值': f'{u_pvalue:.4f}',
            '显著性(p<0.05)': '***' if min(t_pvalue, u_pvalue) < 0.001 else '**' if min(t_pvalue,
                                                                                        u_pvalue) < 0.01 else '*' if min(
                t_pvalue, u_pvalue) < 0.05 else 'NS'
        }
    else:
        # 多组比较：ANOVA和Kruskal-Wallis检验
        f_stat, anova_pvalue = stats.f_oneway(*group_data)
        h_stat, kw_pvalue = stats.kruskal(*group_data)

        means_dict = {f'组{groups[i]}_均值': f'{group_data[i].mean():.4f}' for i in range(len(groups))}

        result = {
            '变量': col,
            **means_dict,
            'ANOVA_p值': f'{anova_pvalue:.4f}',
            'KW检验_p值': f'{kw_pvalue:.4f}',
            '显著性(p<0.05)': '***' if min(anova_pvalue, kw_pvalue) < 0.001 else '**' if min(anova_pvalue,
                                                                                             kw_pvalue) < 0.01 else '*' if min(
                anova_pvalue, kw_pvalue) < 0.05 else 'NS'
        }

    comparison_results.append(result)

# 对离散变量进行卡方检验
print("\n【离散变量组间对比（卡方检验）】")
print("-" * 60)

discrete_results = []
for col in discrete_cols:
    if col == target_variable:
        continue

    # 创建列联表
    contingency_table = pd.crosstab(df[col], df[target_variable])

    # 卡方检验
    chi2, chi_pvalue, dof, expected = stats.chi2_contingency(contingency_table)

    result = {
        '变量': col,
        '卡方统计量': f'{chi2:.4f}',
        'p值': f'{chi_pvalue:.4f}',
        '自由度': dof,
        '显著性(p<0.05)': '***' if chi_pvalue < 0.001 else '**' if chi_pvalue < 0.01 else '*' if chi_pvalue < 0.05 else 'NS'
    }
    discrete_results.append(result)

# 输出结果
print("\n【连续变量对比结果】")
continuous_results_df = pd.DataFrame(comparison_results)
print(continuous_results_df.to_string(index=False))

print("\n【离散变量对比结果】")
discrete_results_df = pd.DataFrame(discrete_results)
print(discrete_results_df.to_string(index=False))

# 保存结果
continuous_results_df.to_excel('连续变量组间对比分析.xlsx', index=False)
discrete_results_df.to_excel('离散变量组间对比分析.xlsx', index=False)

# 筛选显著性变量
significant_continuous = continuous_results_df[continuous_results_df['显著性(p<0.05)'] != 'NS']
significant_discrete = discrete_results_df[discrete_results_df['显著性(p<0.05)'] != 'NS']

print(f"\n【显著性变量汇总】")
print(f"连续变量中有显著性差异的: {len(significant_continuous)} 个")
if len(significant_continuous) > 0:
    print(f"  变量列表: {', '.join(significant_continuous['变量'].tolist())}")

print(f"离散变量中有显著性差异的: {len(significant_discrete)} 个")
if len(significant_discrete) > 0:
    print(f"  变量列表: {', '.join(significant_discrete['变量'].tolist())}")

print("\n分析结果已保存至：")
print("  - 连续变量组间对比分析.xlsx")
print("  - 离散变量组间对比分析.xlsx")
print("=" * 60)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置论文级图表样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 中文显示设置（如需中文标签）
plt.rcParams['axes.unicode_minus'] = False


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 取消注释以显示中文

# ==========================================
# 可视化 1: 连续变量箱线图（带显著性标记）
# ==========================================
def plot_boxplots_with_significance(df, continuous_cols, target_variable, results_df):
    """
    绘制带显著性标记的箱线图
    ★ 改动：按 p 值从小到大排序，只展示最显著的前9个变量（固定 3×3 布局）
    """

    # ══════════════════════════════════════════════════
    # 第1步：筛选显著变量 + 提取数值型p值 + 按p值升序排列
    # ══════════════════════════════════════════════════

    sig_df = results_df[results_df['显著性(p<0.05)'] != 'NS'].copy()

    # 兼容二分类（t检验）和多分类（ANOVA）两种列名
    if 't检验_p值' in sig_df.columns:
        p_col = 't检验_p值'
    elif 'ANOVA_p值' in sig_df.columns:
        p_col = 'ANOVA_p值'
    else:
        print("❌ 结果表中未找到p值列，请检查列名")
        return

    # 强制转为数值型（原来保存时可能是字符串格式如 '0.0023'）
    sig_df[p_col] = pd.to_numeric(sig_df[p_col], errors='coerce')

    # 按 p 值升序排列（p 值越小越显著，排在越前面）
    sig_df = sig_df.sort_values(p_col, ascending=True)

    # 过滤：只保留连续变量 + 排除目标变量本身 + 只取前9个
    all_sig_vars = sig_df['变量'].tolist()
    plot_vars = [
        col for col in all_sig_vars
        if col in continuous_cols and col != target_variable
    ][:9]   # ★ 固定最多9个（3×3）

    if len(plot_vars) == 0:
        print("没有显著性连续变量可绘制")
        return

    # 打印入选变量及其 p 值（便于核查）
    print(f"\n📊 Figure1：显著连续变量共 {len(all_sig_vars)} 个，"
          f"按p值排序后展示前 {len(plot_vars)} 个：")
    for rank, var in enumerate(plot_vars, 1):
        p_row = sig_df[sig_df['变量'] == var]
        p_show = float(p_row[p_col].values[0]) if len(p_row) > 0 else float('nan')
        print(f"  第{rank:>2}名  {var:<30}  p = {p_show:.4f}")

    # ══════════════════════════════════════════════════
    # 第2步：固定 3×3 画布（共9格）
    # ══════════════════════════════════════════════════

    n_vars = len(plot_vars)
    n_cols = 3
    n_rows = 3   # 固定3行，最多容纳9个子图

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.0 * n_rows)   # 13.5 × 12 英寸
    )
    axes = np.array(axes).flatten()   # 始终安全压成一维

    # ══════════════════════════════════════════════════
    # 第3步：颜色配置（支持最多4个分组）
    # ══════════════════════════════════════════════════

    group_palette = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    n_groups      = len(df[target_variable].unique())
    palette       = group_palette[:n_groups]

    # 固定分组顺序（避免 unique() 每次顺序不同）
    groups_sorted = sorted(df[target_variable].dropna().unique())

    # ══════════════════════════════════════════════════
    # 第4步：逐变量绘图
    # ══════════════════════════════════════════════════

    for idx, var in enumerate(plot_vars):
        ax = axes[idx]

        # ── 4a. 箱线图 ─────────────────────────────────
        sns.boxplot(
            x=target_variable, y=var,
            data=df, ax=ax,
            order=groups_sorted,          # 固定组的顺序
            palette=palette,
            width=0.55,
            linewidth=1.2,
            flierprops=dict(
                marker='o', markersize=3,
                markerfacecolor='gray', alpha=0.5
            )
        )

        # ── 4b. 叠加原始数据散点（Strip Plot）────────────
        sns.stripplot(
            x=target_variable, y=var,
            data=df, ax=ax,
            order=groups_sorted,
            color='#2c3e50', alpha=0.25, size=2.5, jitter=True
        )

        # ── 4c. 显著性横线 + 星号标注 ────────────────────
        p_row = sig_df[sig_df['变量'] == var]
        if len(p_row) > 0:
            p_val = float(p_row[p_col].values[0])

            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'

            # ★ 用95%分位数代替max，防止离群值把标注顶飞
            y_high  = df[var].quantile(0.975)
            y_low   = df[var].quantile(0.025)
            y_range = (y_high - y_low) if (y_high != y_low) else 1.0

            # 动态设置 Y 轴范围，上方留出标注空间
            ax.set_ylim([
                y_low  - 0.08 * y_range,
                y_high + 0.30 * y_range
            ])

            bar_y  = y_high + 0.08 * y_range   # 横线高度
            cap_dy = 0.025 * y_range            # 端帽高度

            # 主横线
            ax.plot([0, 1], [bar_y, bar_y], 'k-', lw=1.2)
            # 左端帽
            ax.plot([0, 0], [bar_y - cap_dy, bar_y], 'k-', lw=1.2)
            # 右端帽
            ax.plot([1, 1], [bar_y - cap_dy, bar_y], 'k-', lw=1.2)

            # 显著性符号（横线上方）
            ax.text(
                0.5, bar_y + 0.03 * y_range, sig_text,
                ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='black'
            )

            # 右下角显示具体 p 值（方便审稿人核查）
            p_label = f'p={p_val:.3f}' if p_val >= 0.001 else 'p<0.001'
            ax.text(
                0.97, 0.04, p_label,
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=8, color='#555555', style='italic'
            )

        # ── 4d. 标签与美化 ─────────────────────────────
        ax.set_xlabel('')
        ax.set_ylabel(var, fontsize=10, fontweight='bold')
        ax.set_title(var, fontsize=11, fontweight='bold', pad=6)
        ax.set_xticklabels(
            [f'Group {g}' for g in groups_sorted], fontsize=9
        )
        ax.tick_params(axis='both', labelsize=9, length=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ══════════════════════════════════════════════════
    # 第5步：隐藏多余空白子图（当显著变量 < 9 时）
    # ══════════════════════════════════════════════════

    for idx in range(n_vars, n_rows * n_cols):
        axes[idx].set_visible(False)

    # ══════════════════════════════════════════════════
    # 第6步：总标题 + 图例
    # ══════════════════════════════════════════════════

    plt.suptitle(
        'Top Significant Continuous Variables Between Groups\n'
        '(Ranked by p-value ascending, showing top 9)',
        fontsize=13, fontweight='bold'
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palette[i], label=f'Group {g}', alpha=0.85)
        for i, g in enumerate(groups_sorted)
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=n_groups,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ══════════════════════════════════════════════════
    # 第7步：保存（PNG / PDF / TIFF）
    # ══════════════════════════════════════════════════

    for fmt in ['png', 'pdf', 'tiff']:
        fname = f'Figure1_Boxplots_Top9_by_Pvalue.{fmt}'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  ✅ 已保存: {fname}")

    plt.close()
    print("图1已完成：Figure1_Boxplots_Top9_by_Pvalue.png / .pdf / .tiff\n")


# ==========================================
# 可视化 2: 小提琴图（展示分布形态）
# ==========================================
def plot_violin_plots(df, continuous_cols, target_variable, results_df):
    """绘制小提琴图展示数据分布"""

    sig_vars = results_df[results_df['显著性(p<0.05)'] != 'NS']['变量'].tolist()
    plot_vars = [col for col in sig_vars if col in continuous_cols and col != target_variable][:9]

    if len(plot_vars) == 0:
        return

    n_cols = 3
    n_rows = (len(plot_vars) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if len(plot_vars) > 1 else [axes]

    for idx, var in enumerate(plot_vars):
        ax = axes[idx]

        sns.violinplot(x=target_variable, y=var, data=df, ax=ax,
                       palette=['#3498db', '#e74c3c'], inner='box', cut=0)

        ax.set_xlabel(target_variable)
        ax.set_ylabel(var)
        ax.set_title(f'{var}', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(len(plot_vars), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure2_Violin_Plots.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure2_Violin_Plots.pdf', bbox_inches='tight')
    plt.show()
    print("图2已保存: Figure2_Violin_Plots.png/pdf")


# ==========================================
# 可视化 3: 森林图（展示效应量和置信区间）
# ==========================================
def plot_forest_plot(df, continuous_cols, target_variable, results_df,
                     top_n=20, only_significant=False,
                     out_prefix="Figure3_Forest_Plot_Optimized"):
    """
    Forest plot for continuous variables (binary target)
    Optimizations:
      1) Sort variables by p-value (from results_df if available)
      2) Use Hedges' g (bias-corrected standardized mean difference)
      3) Use Welch t-test for p-values (robust to unequal variance)
      4) Auto xlim padding, grid, larger fonts, better layout
      5) Save png/pdf/tiff
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    import textwrap

    # ─────────────────────────────────────────────
    # 0) 基本检查：二分类目标变量
    # ─────────────────────────────────────────────
    groups = sorted(df[target_variable].dropna().unique())
    if len(groups) != 2:
        print("❌ Forest plot 仅适用于二分类 target（例如 0/1）")
        return

    g0_label, g1_label = groups[0], groups[1]

    # ─────────────────────────────────────────────
    # 1) 候选变量：连续变量且排除 target
    # ─────────────────────────────────────────────
    cand_vars = [c for c in continuous_cols if c != target_variable and c in df.columns]
    if len(cand_vars) == 0:
        print("❌ 没有可用的连续变量")
        return

    # ─────────────────────────────────────────────
    # 2) 从 results_df 获取 p 值（如可用），用于排序
    #    （否则后面会用 Welch t-test 计算）
    # ─────────────────────────────────────────────
    p_col = None
    if results_df is not None and isinstance(results_df, pd.DataFrame):
        if 't检验_p值' in results_df.columns:
            p_col = 't检验_p值'
        elif 'ANOVA_p值' in results_df.columns:
            # 如果你是二分类，理论上应该有 t检验_p值；这里只是兼容
            p_col = 'ANOVA_p值'

    p_map = {}
    if p_col is not None and '变量' in results_df.columns:
        tmp = results_df[['变量', p_col]].copy()
        tmp[p_col] = pd.to_numeric(tmp[p_col], errors='coerce')
        p_map = dict(zip(tmp['变量'], tmp[p_col]))

    # 变量列表按 p 值排序（没有 p 值的排后）
    def _p_for_sort(v):
        pv = p_map.get(v, np.nan)
        return pv if pd.notna(pv) else 1e9

    cand_vars = sorted(cand_vars, key=_p_for_sort)

    # 如果只画显著
    if only_significant and len(p_map) > 0:
        cand_vars = [v for v in cand_vars if pd.notna(p_map.get(v, np.nan)) and p_map[v] < 0.05]

    # 限制数量
    cand_vars = cand_vars[:top_n]
    if len(cand_vars) == 0:
        print("没有满足条件的变量可绘制森林图")
        return

    # ─────────────────────────────────────────────
    # 3) 计算：Hedges' g + 95%CI + Welch p-value
    # ─────────────────────────────────────────────
    rows = []

    for var in cand_vars:
        a = df.loc[df[target_variable] == g0_label, var].dropna().astype(float)
        b = df.loc[df[target_variable] == g1_label, var].dropna().astype(float)

        n0, n1 = len(a), len(b)
        if n0 < 3 or n1 < 3:
            continue

        mean0, mean1 = a.mean(), b.mean()
        sd0, sd1 = a.std(ddof=1), b.std(ddof=1)

        # pooled SD
        denom = (n0 + n1 - 2)
        if denom <= 0:
            continue
        pooled_var = ((n0 - 1) * sd0**2 + (n1 - 1) * sd1**2) / denom
        pooled_sd = np.sqrt(pooled_var)

        if pooled_sd == 0 or np.isnan(pooled_sd):
            continue

        # Cohen's d
        d = (mean1 - mean0) / pooled_sd

        # Hedges' g correction (small sample bias)
        J = 1 - (3 / (4 * (n0 + n1) - 9))   # 常用近似
        g = J * d

        # Approx SE and 95%CI for g
        # (近似公式，医学文章常用；样本很小可考虑bootstrap)
        se_g = np.sqrt((n0 + n1) / (n0 * n1) + (g**2) / (2 * (n0 + n1 - 2)))
        ci_l = g - 1.96 * se_g
        ci_u = g + 1.96 * se_g

        # Welch t-test p value
        _, p_welch = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')

        # 若 results_df 里有 p 值，用它作为“展示/排序p”（可选）
        p_show = p_map.get(var, np.nan)
        if pd.isna(p_show):
            p_show = p_welch

        rows.append({
            "变量": var,
            "g": g,
            "ci_l": ci_l,
            "ci_u": ci_u,
            "p": float(p_show),
            "p_welch": float(p_welch),
            "n0": n0,
            "n1": n1,
            "mean0": mean0, "sd0": sd0,
            "mean1": mean1, "sd1": sd1,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("❌ 可绘制的变量为空（可能样本量过小或标准差为0）")
        return

    # 再按 p 排序（更稳）
    res = res.sort_values("p", ascending=True).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # 4) 画图参数（字体更大、网格、自动x范围）
    # ─────────────────────────────────────────────
    FS_TITLE = 22
    FS_AXIS  = 16
    FS_TICK  = 15
    FS_ANN   = 15

    n_vars = len(res)
    fig_h = max(6, 0.45 * n_vars + 1.5)
    fig_w = 12

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # y 轴从上到下：p最小在最上方（更符合读者习惯）
    y = np.arange(n_vars)[::-1]
    res_plot = res.iloc[::-1].reset_index(drop=True)

    # 自动 x 范围 + padding
    x_min = res["ci_l"].min()
    x_max = res["ci_u"].max()
    pad = 0.15 * (x_max - x_min) if x_max > x_min else 1.0
    ax.set_xlim(x_min - pad, x_max + 3.0 * pad)  # 右侧留空间写p值

    # 0线 + 网格
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    # 绘制误差棒
    for i in range(n_vars):
        g = res_plot.loc[i, "g"]
        cl = res_plot.loc[i, "ci_l"]
        cu = res_plot.loc[i, "ci_u"]
        p = res_plot.loc[i, "p"]

        color = "#D63031" if p < 0.05 else "#7F8C8D"
        ax.errorbar(
            g, y[i],
            xerr=[[g - cl], [cu - g]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=4,
            markersize=8
        )

    # y tick label：变量名过长自动换行
    y_labels = [
        "\n".join(textwrap.wrap(v, width=26))
        for v in res_plot["变量"].tolist()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=FS_TICK)

    ax.set_xlabel("Hedges' g (95% CI)", fontsize=FS_AXIS, fontweight="bold")
    ax.set_title("Forest Plot of Standardized Mean Difference", fontsize=FS_TITLE, fontweight="bold", pad=12)

    # 右侧标注 p 和样本量
    x_text = x_max + 1.6 * pad
    for i in range(n_vars):
        p = res_plot.loc[i, "p"]
        n0 = int(res_plot.loc[i, "n0"])
        n1 = int(res_plot.loc[i, "n1"])

        if p < 0.001:
            p_txt = "p<0.001"
            star = "***"
        elif p < 0.01:
            p_txt = f"p={p:.3f}"
            star = "**"
        elif p < 0.05:
            p_txt = f"p={p:.3f}"
            star = "*"
        else:
            p_txt = f"p={p:.3f}"
            star = ""

        ax.text(
            x_text, y[i],
            f"{p_txt} {star}   (n={n0}/{n1})",
            va="center",
            fontsize=FS_ANN,
            color="#2c3e50"
        )

    # 去掉多余边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # ─────────────────────────────────────────────
    # 5) 保存
    # ─────────────────────────────────────────────
    for ext in ["png", "pdf", "tiff"]:
        fn = f"{out_prefix}.{ext}"
        plt.savefig(fn, dpi=300, bbox_inches="tight")
        print(f"  ✅ 已保存: {fn}")

    plt.close()
    print("Figure3 优化版森林图生成完成。\n")




# ==========================================
# 可视化 4: 离散变量堆叠柱状图
# ==========================================
def plot_stacked_bar(df, discrete_cols, target_variable, results_df):
    """绘制离散变量堆叠柱状图"""

    sig_vars = results_df[results_df['显著性(p<0.05)'] != 'NS']['变量'].tolist()
    plot_vars = [col for col in sig_vars if col in discrete_cols and col != target_variable][:6]

    if len(plot_vars) == 0:
        print("没有显著性离散变量")
        return

    n_cols = 2
    n_rows = (len(plot_vars) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten() if len(plot_vars) > 1 else [axes]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, var in enumerate(plot_vars):
        ax = axes[idx]

        # 计算交叉表百分比
        ct = pd.crosstab(df[var], df[target_variable], normalize='index') * 100

        ct.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(ct.columns)],
                edgecolor='white', width=0.7)

        ax.set_xlabel(var, fontweight='bold')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{var}', fontweight='bold')
        ax.legend(title=target_variable, bbox_to_anchor=(1.02, 1))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # 获取p值
        row = results_df[results_df['变量'] == var]
        if len(row) > 0:
            p_val = float(row['p值'].values[0])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            ax.text(0.95, 0.95, f'p={p_val:.3f}{sig}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(len(plot_vars), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('Figure4_Stacked_Bar_Charts.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figure4_Stacked_Bar_Charts.pdf', bbox_inches='tight')
    plt.show()
    print("图4已保存: Figure4_Stacked_Bar_Charts.png/pdf")


# ==========================================
# 可视化 5: 热力图（相关性 + 显著性）
# ==========================================
def plot_correlation_heatmap(df, continuous_cols, target_variable):
    """
    绘制相关性热力图
    ★ 字号改动：
       - 热图格子内注释字号：8  → 13
       - 坐标轴刻度标签字号：9  → 13
       - 颜色条标签字号    ：默认→ 13
       - 标题字号          ：12 → 15
       - 对角线注释字号    ：同上格子注释
    """

    # ══════════════════════════════════════════════════
    # ★ 字号控制区（只改这里即可全局调整）
    # ══════════════════════════════════════════════════
    FS_ANNOT  = 18    # 热图格子内文字（r值 / p值 / 星号）
    FS_TICK   = 20    # X / Y 轴刻度标签
    FS_CBAR   = 20    # 颜色条标签
    FS_TITLE  = 25    # 总标题

    # ══════════════════════════════════════════════════
    # 第1步：准备变量列表
    # ══════════════════════════════════════════════════
    corr_cols = [
        col for col in continuous_cols
        if col != target_variable
    ][:15]

    if len(corr_cols) < 2:
        print("连续变量不足2个，无法绘制热力图")
        return

    print(f"\n📊 Figure5：热力图包含 {len(corr_cols)} 个连续变量")

    # ══════════════════════════════════════════════════
    # 第2步：计算 Pearson 相关矩阵
    # ══════════════════════════════════════════════════
    corr_matrix = df[corr_cols].corr()

    # ══════════════════════════════════════════════════
    # 第3步：计算 p 值矩阵（修复不等长 bug）
    # ══════════════════════════════════════════════════
    n = len(corr_cols)
    p_matrix = pd.DataFrame(
        np.ones((n, n)),
        index=corr_cols,
        columns=corr_cols
    )

    for i, col1 in enumerate(corr_cols):
        for j, col2 in enumerate(corr_cols):
            if i == j:
                p_matrix.loc[col1, col2] = 0.0001   # 对角线自相关
            else:
                pair_data = df[[col1, col2]].dropna()
                if len(pair_data) > 2:
                    _, p = stats.pearsonr(
                        pair_data[col1], pair_data[col2]
                    )
                    p_matrix.loc[col1, col2] = p
                else:
                    p_matrix.loc[col1, col2] = 1.0

    # ══════════════════════════════════════════════════
    # 第4步：构建注释矩阵（r值 + p值 + 星号，三行显示）
    # ══════════════════════════════════════════════════
    annot_matrix = pd.DataFrame(
        '',
        index=corr_cols,
        columns=corr_cols
    )

    for i, col1 in enumerate(corr_cols):
        for j, col2 in enumerate(corr_cols):

            r_val = corr_matrix.loc[col1, col2]
            p_val = p_matrix.loc[col1, col2]

            # 显著性星号
            if p_val < 0.001:
                star = '***'
            elif p_val < 0.01:
                star = '**'
            elif p_val < 0.05:
                star = '*'
            else:
                star = 'ns'

            # p 值文字
            p_text = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'

            if i == j:
                # 对角线三行：r值 / p值 / 星号
                annot_matrix.loc[col1, col2] = (
                    f'1.00\n{p_text}\n{star}'
                )
            else:
                # 下三角三行
                annot_matrix.loc[col1, col2] = (
                    f'{r_val:.2f}\n{p_text}\n{star}'
                )

    # ══════════════════════════════════════════════════
    # 第5步：掩码（k=1：只掩盖严格上三角，对角线显示）
    # ══════════════════════════════════════════════════
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # ══════════════════════════════════════════════════
    # 第6步：画布尺寸（格子内文字变大 → 画布相应放大）
    # ══════════════════════════════════════════════════
    # 每个格子需要容纳三行大字，单元格至少 1.5 英寸
    cell_inch = max(1.5, 18 / n)
    fig_w     = n * cell_inch + 2.5    # 右侧留颜色条空间
    fig_h     = n * cell_inch + 1.5    # 顶部留标题空间

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ══════════════════════════════════════════════════
    # 第7步：绘制热力图（★ annot_kws 字号放大）
    # ══════════════════════════════════════════════════
    hm = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot_matrix,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.6,
        linecolor='#cccccc',
        ax=ax,
        cbar_kws={
            'shrink'     : 0.70,
            'pad'        : 0.02,
            'label'      : 'Pearson r',
        },
        annot_kws={
            'size'   : FS_ANNOT,     # ★ 格子内文字字号
            'weight' : 'normal',
            'va'     : 'center',
            'linespacing' : 1.4,     # 三行文字的行间距
        }
    )

    # ══════════════════════════════════════════════════
    # 第8步：颜色条字号放大
    # ══════════════════════════════════════════════════
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FS_CBAR)          # 刻度数字
    cbar.set_label('Pearson r', fontsize=FS_CBAR)   # 颜色条标题

    # ══════════════════════════════════════════════════
    # 第9步：对角线格子金色背景高亮
    # ══════════════════════════════════════════════════
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1,
            fill=True,
            facecolor='#FFD700',
            alpha=0.28,
            zorder=2
        ))

    # ══════════════════════════════════════════════════
    # 第10步：坐标轴标签字号放大
    # ══════════════════════════════════════════════════
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45, ha='right',
        fontsize=FS_TICK,              # ★ X轴刻度字号
        fontweight='bold'
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=FS_TICK,              # ★ Y轴刻度字号
        fontweight='bold'
    )

    # ══════════════════════════════════════════════════
    # 第11步：标题（★ 字号放大）
    # ══════════════════════════════════════════════════
    ax.set_title(
        'Pearson Correlation Heatmap with P-values\n'
        '( *p<0.05   **p<0.01   ***p<0.001   ns: not significant )\n'
        '★ Diagonal (e.g., Age–Age, PCT–PCT): Self-correlation = 1.00, p<0.001 ***',
        fontweight='bold',
        fontsize=FS_TITLE,             # ★ 标题字号
        pad=20
    )

    plt.tight_layout()

    # ══════════════════════════════════════════════════
    # 第12步：保存
    # ══════════════════════════════════════════════════
    for fmt in ['png', 'pdf']:
        fname = f'Figure5_Correlation_Heatmap.{fmt}'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  ✅ 已保存: {fname}")

    plt.close()
    print("图5已完成：Figure5_Correlation_Heatmap.png / .pdf\n")


# ==========================================
# 可视化 6: 综合对比图（论文主图）
# ==========================================
def plot_publication_summary(df, continuous_cols,
                             target_variable, continuous_results_df):
    """
    综合对比图（小提琴图 + 箱线图叠加），p值前9，3×3布局
    ★ 颜色改动：
       Group 0 → 蓝色系
       Group 1 → 红色系
    """
    import numpy as np
    from matplotlib.patches import Patch
    import matplotlib.cm as cm

    # ══════════════════════════════════════════════════
    # ★ 字号控制区
    # ══════════════════════════════════════════════════
    FS_TITLE    = 20
    FS_SUBTITLE = 18
    FS_YLABEL   = 15
    FS_XTICK    = 15
    FS_YTICK    = 15
    FS_SIG      = 18
    FS_PVAL     = 14
    FS_LEGEND   = 15

    # ══════════════════════════════════════════════════
    # ★ 颜色控制区（只改这里即可调整配色）
    # ══════════════════════════════════════════════════
    #   violin_colors : 小提琴填充色（半透明背景层）
    #   box_colors    : 箱线图填充色（不透明前景层）
    #
    #   索引 0 → Group 0 → 蓝色系
    #   索引 1 → Group 1 → 红色系
    # ──────────────────────────────────────────────────
    violin_colors = [
        '#74B9FF',   # Group 0：浅蓝（小提琴）
        '#FF7675',   # Group 1：浅红（小提琴）
    ]
    box_colors = [
        '#0984E3',   # Group 0：深蓝（箱线图）
        '#D63031',   # Group 1：深红（箱线图）
    ]

    # ══════════════════════════════════════════════════
    # 第1步：筛选显著变量 + 按 p 值升序 + 只取前9
    # ══════════════════════════════════════════════════
    sig_df = continuous_results_df[
        continuous_results_df['显著性(p<0.05)'] != 'NS'
    ].copy()

    if 't检验_p值' in sig_df.columns:
        p_col = 't检验_p值'
    elif 'ANOVA_p值' in sig_df.columns:
        p_col = 'ANOVA_p值'
    else:
        print("❌ 未找到p值列，请检查continuous_results_df的列名")
        return

    sig_df[p_col] = pd.to_numeric(sig_df[p_col], errors='coerce')
    sig_df = sig_df.sort_values(p_col, ascending=True)

    top_vars = [
        col for col in sig_df['变量'].tolist()
        if col in continuous_cols and col != target_variable
    ][:9]

    if len(top_vars) == 0:
        print("没有显著连续变量可绘制")
        return

    print(f"\n📊 Figure6：按p值排序后展示前 {len(top_vars)} 个显著变量：")
    print(f"  {'排名':<6}{'变量名':<35}{'p 值'}")
    print(f"  {'-'*6}{'-'*35}{'-'*12}")
    for rank, var in enumerate(top_vars, 1):
        p_row  = sig_df[sig_df['变量'] == var]
        p_show = float(p_row[p_col].values[0]) if len(p_row) > 0 else float('nan')
        print(f"  {rank:<6}{var:<35}p = {p_show:.4f}")

    # ══════════════════════════════════════════════════
    # 第2步：固定 3×3 画布
    # ══════════════════════════════════════════════════
    n_vars = len(top_vars)
    n_cols = 3
    n_rows = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 5.2 * n_rows)
    )
    axes = np.array(axes).flatten()

    # ══════════════════════════════════════════════════
    # 第3步：分组配置
    # ══════════════════════════════════════════════════
    groups_sorted = sorted(df[target_variable].dropna().unique())
    n_groups      = len(groups_sorted)

    # ★ 如颜色列表不够用（超过2组），自动补充备用色
    extra_violin = ['#55EFC4', '#FDCB6E', '#A29BFE']
    extra_box    = ['#00B894', '#E17055', '#6C5CE7']
    while len(violin_colors) < n_groups:
        violin_colors.append(extra_violin[len(violin_colors) - 2])
        box_colors.append(extra_box[len(box_colors) - 2])

    # X轴标签（按实际含义修改，例如 {0:'Non-PE', 1:'PE'}）
    group_labels = {g: f'Group {g}' for g in groups_sorted}

    # ══════════════════════════════════════════════════
    # 第4步：逐变量绘图
    # ══════════════════════════════════════════════════
    for idx, var in enumerate(top_vars):
        ax = axes[idx]

        group_data = [
            df[df[target_variable] == g][var].dropna().values
            for g in groups_sorted
        ]

        # ── 4a. 小提琴图（半透明背景层）──────────────────
        parts = ax.violinplot(
            group_data,
            positions=range(n_groups),
            showmeans=False,
            showmedians=False
        )
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors[i])   # ★ 蓝/红
            pc.set_edgecolor(box_colors[i])       # ★ 描边用深色
            pc.set_linewidth(0.8)
            pc.set_alpha(0.35)

        for key in ['cbars', 'cmins', 'cmaxes']:
            if key in parts:
                parts[key].set_visible(False)

        # ── 4b. 箱线图（不透明前景层）──────────────────────
        bp = ax.boxplot(
            group_data,
            positions=range(n_groups),
            widths=0.30,
            patch_artist=True,
            medianprops=dict(color='white', linewidth=2.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(
                marker='o', markersize=4,
                markerfacecolor='gray', alpha=0.4
            )
        )
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(box_colors[i])   # ★ 深蓝/深红
            patch.set_alpha(0.88)

        # 须线和端帽也染色（★ 增强视觉统一感）
        n_each = n_groups   # 每组的须线数 = 2，端帽数 = 2
        for i in range(n_groups):
            for j in range(2):   # 每组2条须线、2个端帽
                bp['whiskers'][i * 2 + j].set_color(box_colors[i])
                bp['caps'][i * 2 + j].set_color(box_colors[i])
            bp['medians'][i].set_color('white')

        # ── 4c. Y 轴范围（百分位数防离群值顶飞）──────────
        all_vals = pd.concat([
            df[df[target_variable] == g][var].dropna()
            for g in groups_sorted
        ])
        y_low   = all_vals.quantile(0.025)
        y_high  = all_vals.quantile(0.975)
        y_range = (y_high - y_low) if (y_high != y_low) else 1.0

        ax.set_ylim([
            y_low  - 0.10 * y_range,
            y_high + 0.42 * y_range
        ])

        # ── 4d. 显著性标注 ─────────────────────────────────
        p_row = sig_df[sig_df['变量'] == var]
        if len(p_row) > 0:
            p_val = float(p_row[p_col].values[0])

            sig_text = ('***' if p_val < 0.001 else
                        '**'  if p_val < 0.01  else
                        '*'   if p_val < 0.05  else 'ns')
            p_label  = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'

            bar_y  = y_high + 0.10 * y_range
            cap_dy = 0.030 * y_range
            star_y = bar_y  + 0.04 * y_range
            pval_y = star_y + 0.10 * y_range

            ax.plot([0, n_groups - 1], [bar_y, bar_y],
                    'k-', lw=1.8)
            ax.plot([0, 0],
                    [bar_y - cap_dy, bar_y], 'k-', lw=1.8)
            ax.plot([n_groups - 1, n_groups - 1],
                    [bar_y - cap_dy, bar_y], 'k-', lw=1.8)

            ax.text(
                (n_groups - 1) / 2, star_y, sig_text,
                ha='center', va='bottom',
                fontsize=FS_SIG, fontweight='bold', color='black'
            )
            ax.text(
                (n_groups - 1) / 2, pval_y, p_label,
                ha='center', va='bottom',
                fontsize=FS_PVAL, color='#333333', style='italic'
            )

        # ── 4e. 坐标轴美化 ─────────────────────────────────
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(
            [group_labels[g] for g in groups_sorted],
            fontsize=FS_XTICK, fontweight='bold'
        )

        # ★ X 轴刻度标签颜色：Group 0 蓝 / Group 1 红
        for tick_label, g in zip(ax.get_xticklabels(), groups_sorted):
            gi = list(groups_sorted).index(g)
            tick_label.set_color(box_colors[gi])   # ★

        ax.set_ylabel(var, fontweight='bold', fontsize=FS_YLABEL)
        ax.set_title(var, fontsize=FS_SUBTITLE, fontweight='bold', pad=8)
        ax.tick_params(axis='y', labelsize=FS_YTICK)
        ax.tick_params(axis='x', length=4, labelsize=FS_XTICK)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ══════════════════════════════════════════════════
    # 第5步：隐藏多余空白子图
    # ══════════════════════════════════════════════════
    for idx in range(n_vars, n_rows * n_cols):
        axes[idx].set_visible(False)

    # ══════════════════════════════════════════════════
    # 第6步：图例（★ Group 0 蓝 / Group 1 红）
    # ══════════════════════════════════════════════════
    legend_elements = [
        Patch(
            facecolor=box_colors[i],
            edgecolor=box_colors[i],
            label=group_labels[g],
            alpha=0.88
        )
        for i, g in enumerate(groups_sorted)
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=n_groups,
        frameon=False,
        fontsize=FS_LEGEND
    )

    # ══════════════════════════════════════════════════
    # 第7步：总标题
    # ══════════════════════════════════════════════════
    plt.suptitle(
        'Top 9 Significant Variables Between Groups\n'
        '(Ranked by p-value ascending)',
        fontsize=FS_TITLE, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    # ══════════════════════════════════════════════════
    # 第8步：保存
    # ══════════════════════════════════════════════════
    for fmt in ['png', 'pdf', 'tiff']:
        fname = f'Figure6_Publication_Summary_Top9.{fmt}'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  ✅ 已保存: {fname}")

    plt.close()
    print("图6已完成：Figure6_Publication_Summary_Top9.png / .pdf / .tiff\n")


# ==========================================
# 执行所有可视化
# ==========================================
print("\n" + "=" * 60)
print("开始生成论文级可视化图表")
print("=" * 60)

# Figure 1：p值排名前9箱线图（3×3）
plot_boxplots_with_significance(
    df, continuous_cols, target_variable, continuous_results_df
)

# Figure 2：小提琴图
plot_violin_plots(
    df, continuous_cols, target_variable, continuous_results_df
)

# Figure 3：森林图
plot_forest_plot(
    df, continuous_cols, target_variable, continuous_results_df
)

# Figure 4：离散变量堆叠柱状图
plot_stacked_bar(
    df, discrete_cols, target_variable, discrete_results_df
)

# Figure 5：相关性热力图（对角线含 p 值）
plot_correlation_heatmap(
    df, continuous_cols, target_variable
)

# Figure 6：p值排名前9综合对比图（3×3）
plot_publication_summary(
    df, continuous_cols, target_variable, continuous_results_df
)

print("\n" + "=" * 60)
print("✅ 所有图表生成完成！")
print("=" * 60)
print("\n生成的文件清单：")
files = [
    ("Figure1", "Figure1_Boxplots_Top9_by_Pvalue",       "png / pdf / tiff", "p值前9，3×3箱线图"),
    ("Figure2", "Figure2_Violin_Plots",                   "png / pdf",        "小提琴图"),
    ("Figure3", "Figure3_Forest_Plot",                    "png / pdf",        "森林图"),
    ("Figure4", "Figure4_Stacked_Bar_Charts",             "png / pdf",        "离散变量堆叠柱状图"),
    ("Figure5", "Figure5_Correlation_Heatmap",            "png / pdf",        "热力图（含对角线p值）"),
    ("Figure6", "Figure6_Publication_Summary_Top9",       "png / pdf / tiff", "p值前9，3×3综合对比图"),
]
for fig_no, fname, fmts, desc in files:
    print(f"  • {fig_no}: {fname}.{{{fmts}}}  ← {desc}")
