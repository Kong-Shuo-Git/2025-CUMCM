import warnings
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from math import sin, sqrt, pi
from matplotlib.patches import Patch
import os
from datetime import datetime
from scipy import stats

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
# 增加全局字体大小：从12改为14
plt.rcParams['font.size'] = 14
plt.rcParams['font.size'] = 12  # 缩小字体，避免图表拥挤
color1 = "#00BA38"  # 极大值点
color2 = "#619CFF"  # 极小值点
color3 = "#C86193"  # 厚度分布
color4 = "#F8766D"  # 残差分析
grid_color = 'midgray'
text_color = 'black'
bg_color = 'lightgray'

# 确保输出目录存在
os.makedirs('output', exist_ok=True)
os.makedirs('output/images', exist_ok=True)
os.makedirs('output/excel', exist_ok=True)
os.makedirs('output/txt', exist_ok=True)
os.makedirs('output/stability', exist_ok=True)


# -------------------------- 1. 修正物理模型参数（碳化硅标准参数） --------------------------
def cauchy_refractive_index(lambda_μm, A=2.65, B=0.015, C=1e-7):
    """Cauchy模型：适配碳化硅红外波段（2.5-5μm）"""
    n = A + B / (lambda_μm ** 2) + C / (lambda_μm ** 4)
    return n


def sellmeier_refractive_index(lambda_μm, A1=6.91, B1=0.202):
    """Sellmeier模型：碳化硅标准参数（λ² > B1 μm²，B1=0.202对应λ>0.45μm，满足2.5-5μm范围）"""
    if lambda_μm ** 2 <= B1:
        raise ValueError(f"波长 {lambda_μm:.2f}μm 过小，需λ² > {B1}μm²")
    n_squared = 1 + (A1 * lambda_μm ** 2) / (lambda_μm ** 2 - B1)
    return sqrt(n_squared)


# -------------------------- 2. 数据预处理：适配实际波长范围 --------------------------
def preprocess_data(file_path, angle):
    try:
        data = pd.read_excel(file_path)
    except Exception:
        data = pd.read_csv(file_path)

    # 数据清洗与波长转换（波数cm⁻¹ → 波长nm：λ(nm)=1e7/波数(cm⁻¹)）
    data.columns = ["波数 (cm-1)", "反射率 (%)"]
    data["波数 (cm-1)"] = pd.to_numeric(data["波数 (cm-1)"], errors='coerce')
    data["反射率 (%)"] = pd.to_numeric(data["反射率 (%)"], errors='coerce')
    data = data.dropna()
    data["lambda_nm"] = 1e7 / data["波数 (cm-1)"]

    # 修正波长范围：附件数据实际为2500-5000nm（波数2000-4000cm⁻¹），避免无效数据
    valid_mask = (data["lambda_nm"] >= 2500) & (data["lambda_nm"] <= 5000)
    data_valid = data[valid_mask].sort_values("lambda_nm", ascending=False).reset_index(drop=True)

    # 增加数据量判断（降低阈值，避免误判）
    if len(data_valid) < 50:
        raise ValueError(f"有效数据点仅{len(data_valid)}个，需≥50个")

    data_valid["incident_angle"] = angle
    print(
        f"预处理后数据：波长范围{data_valid['lambda_nm'].min():.0f}-{data_valid['lambda_nm'].max():.0f}nm，共{len(data_valid)}个点")
    return data_valid


# -------------------------- 3. 优化极值点检测（降低阈值，增加峰显著性判断） --------------------------
def detect_interference_extrema(data_valid):
    R = data_valid["反射率 (%)"].values
    lambda_nm = data_valid["lambda_nm"].values
    r_mean = np.mean(R)
    r_std = np.std(R)

    # 优化阈值：降低高度门槛，增加峰显著性（prominence）过滤噪声
    min_height_max = r_mean + 0.2 * r_std  # 极大值最小高度（原代码仅用min_height，此处分开设置更灵活）
    min_height_min = r_mean - 0.8 * r_std  # 极小值最小高度（降低门槛，避免漏检）
    min_distance = max(10, int(len(data_valid) * 0.02))  # 最小峰间距：至少10个点，避免过疏
    prominence = 0.05  # 峰显著性：过滤微小波动（反射率变化≥0.05%才视为峰）

    # 检测极大值（峰）和极小值（谷）
    peak_indices, _ = find_peaks(
        R,
        distance=min_distance,
        height=min_height_max,
        prominence=prominence
    )
    valley_indices, _ = find_peaks(
        -R,  # 极小值=负反射率的极大值
        distance=min_distance,
        height=-min_height_min,  # 对应原反射率的极小值高度
        prominence=prominence
    )

    # 整合极值点
    extrema = pd.concat([
        pd.DataFrame({"type": "max", "lambda_nm": lambda_nm[peak_indices], "反射率 (%)": R[peak_indices]}),
        pd.DataFrame({"type": "min", "lambda_nm": lambda_nm[valley_indices], "反射率 (%)": R[valley_indices]})
    ]).sort_values("lambda_nm", ascending=False).reset_index(drop=True)

    # 打印极值点数量，方便调试
    print(f"检测到极值点：共{len(extrema)}个（极大值{len(peak_indices)}个，极小值{len(valley_indices)}个）")
    if len(extrema) < 3:
        raise ValueError(f"极值点不足3个，需调整检测参数")

    return extrema


# -------------------------- 4. 优化厚度计算（增加异常值过滤） --------------------------
def calculate_thickness(extrema, incident_angle, ref_model, **model_params):
    if len(extrema) < 2:
        return pd.DataFrame(), np.nan, np.nan, np.nan, pd.DataFrame()

    ref_index_func = cauchy_refractive_index if ref_model == "cauchy" else sellmeier_refractive_index
    angle_rad = incident_angle * pi / 180
    lambda0_nm = extrema.iloc[0]["lambda_nm"]
    k0_estimates = []

    # 计算k0（干涉级次基数）：过滤过小的波长差，避免异常值
    for i in range(1, len(extrema)):
        m_i = i * 0.5  # 干涉级次差（峰-谷/谷-峰差0.5级）
        lambda_i_nm = extrema.iloc[i]["lambda_nm"]
        lambda_diff = lambda0_nm - lambda_i_nm
        if lambda_diff > 10:  # 波长差≥10nm才计算，避免除以微小值
            k0_est = (m_i * lambda_i_nm) / lambda_diff
            k0_estimates.append(k0_est)

    if not k0_estimates:
        print("无有效k0估算值，需调整波长差阈值")
        return pd.DataFrame(), np.nan, np.nan, np.nan, pd.DataFrame()

    k0_final = np.median(k0_estimates)  # 用中位数抗异常值
    results = []
    residuals = []
    expected_k = []
    actual_k = []

    # 计算厚度并过滤异常值（厚度为正且在合理范围：0.1-100μm）
    for i, row in extrema.iterrows():
        m_i = i * 0.5
        k_i = k0_final + m_i
        lambda_i_nm = row["lambda_nm"]
        lambda_i_μm = lambda_i_nm / 1000.0

        # 计算折射率
        try:
            n_i = ref_index_func(lambda_i_μm, **model_params)
        except Exception as e:
            print(f"计算折射率失败：{e}，跳过该点")
            continue

        # 计算厚度（干涉公式：2ndcosθ = kλ → d = kλ/(2ncosθ)，cosθ=√(n²-sin²θ_incident)）
        denominator = 2 * sqrt(n_i ** 2 - sin(angle_rad) ** 2)
        thickness = (k_i * lambda_i_μm) / denominator

        # 过滤异常厚度（碳化硅外延层常见厚度0.5-50μm）
        if 0.1 < thickness < 100:
            results.append({
                "lambda_nm": lambda_i_nm,
                "thickness_μm": thickness,
                "k_i": k_i,
                "n_i": n_i
            })
            # 计算残差（验证模型一致性）
            expected_k_val = (2 * thickness * denominator) / lambda_i_μm
            residuals.append(k_i - expected_k_val)
            expected_k.append(expected_k_val)
            actual_k.append(k_i)

    thickness_df = pd.DataFrame(results)
    if thickness_df.empty:
        print("无有效厚度值，需调整厚度过滤范围")
        return thickness_df, np.nan, np.nan, np.nan, pd.DataFrame()

    # 计算厚度统计量
    avg_thickness = thickness_df["thickness_μm"].mean()
    std_thickness = thickness_df["thickness_μm"].std()
    rsd = (std_thickness / avg_thickness) * 100 if avg_thickness != 0 else np.inf

    # 残差数据
    residual_data = pd.DataFrame({
        "lambda_nm": thickness_df["lambda_nm"],
        "actual_k": actual_k,
        "expected_k": expected_k,
        "residual": residuals,
        "thickness_μm": thickness_df["thickness_μm"]
    })

    print(f"{ref_model}模型：平均厚度{avg_thickness:.4f}μm，RSD={rsd:.2f}%")
    return thickness_df, avg_thickness, std_thickness, rsd, residual_data


# -------------------------- 5. 优化子区间搜索（适配实际波长范围） --------------------------
def find_best_bands(data_valid, incident_angle, model_name, model_params, window_size=800, step_size=200,
                    min_extrema=3):
    """
    window_size：子区间宽度（800nm，适配2500-5000nm范围，可生成多个子区间）
    step_size：步长（200nm，增加子区间数量）
    min_extrema：每个子区间至少3个极值点（降低门槛）
    """
    all_bands_results = []
    min_lambda = data_valid["lambda_nm"].min()
    max_lambda = data_valid["lambda_nm"].max()
    ref_index_func = cauchy_refractive_index if model_name == "cauchy" else sellmeier_refractive_index

    # 生成子区间（确保不超出数据范围）
    for start_lambda in np.arange(min_lambda, max_lambda - window_size + 100, step_size):
        end_lambda = start_lambda + window_size
        band_data = data_valid[(data_valid["lambda_nm"] >= start_lambda) & (data_valid["lambda_nm"] <= end_lambda)]

        # 子区间数据量≥30个点才分析
        if len(band_data) < 30:
            continue

        try:
            extrema = detect_interference_extrema(band_data)
            if len(extrema) < min_extrema:
                continue

            # 计算该区间厚度
            _, avg_thick, _, rsd, residuals = calculate_thickness(extrema, incident_angle, model_name, **model_params)

            # 过滤合理结果（RSD<50%，残差非空）
            if pd.notna(rsd) and rsd < 50 and residuals is not None and not residuals.empty:
                # 计算折射率范围
                lambda_min_μm = start_lambda / 1000.0
                lambda_max_μm = end_lambda / 1000.0
                n_min = ref_index_func(lambda_max_μm, **model_params)  # 波长越大，折射率越小
                n_max = ref_index_func(lambda_min_μm, **model_params)

                all_bands_results.append({
                    "start_lambda": start_lambda,
                    "end_lambda": end_lambda,
                    "avg_thickness": avg_thick,
                    "rsd": rsd,
                    "extrema_count": len(extrema),
                    "n_min": n_min,
                    "n_max": n_max,
                    "residual_std": residuals["residual"].std(),
                    "residual_mean": residuals["residual"].mean(),
                    "residual_rmse": np.sqrt(np.mean(residuals["residual"] ** 2))
                })
        except Exception as e:
            print(f"子区间[{start_lambda:.0f}-{end_lambda:.0f}nm]分析失败：{str(e)[:50]}")
            continue

    bands_df = pd.DataFrame(all_bands_results)
    print(f"{model_name}模型：找到{len(bands_df)}个有效子区间")
    return bands_df


# -------------------------- 6. 优化加权厚度计算（降低筛选门槛） --------------------------
def get_weighted_thickness(band_results_df, rsd_threshold=30.0, top_n=3):
    """降低RSD阈值（30%），确保有足够区间参与加权"""
    if band_results_df.empty:
        return np.nan, None

    # 筛选优良区间（RSD<30%且残差RMSE<0.5）
    good_bands = band_results_df[
        (band_results_df['rsd'] < rsd_threshold) &
        (band_results_df['residual_rmse'] < 0.5)
        ].copy()

    # 若无优良区间，取综合得分前3的区间
    if good_bands.empty:
        band_results_df['combined_score'] = band_results_df['rsd'] * 0.6 + band_results_df['residual_rmse'] * 0.4
        good_bands = band_results_df.sort_values('combined_score').head(top_n).copy()
        print(f"无满足RSD<{rsd_threshold}%的区间，取综合得分前{top_n}的区间")

    # 计算权重（基于RSD和残差，避免除以零）
    good_bands['weight_rsd'] = 1 / (good_bands['rsd'] + 1e-6)
    good_bands['weight_residual'] = 1 / (good_bands['residual_rmse'] + 1e-6)
    good_bands['weight'] = (good_bands['weight_rsd'] + good_bands['weight_residual']) / 2
    total_weight = good_bands['weight'].sum()
    good_bands['norm_weight'] = good_bands['weight'] / total_weight

    # 加权平均厚度
    weighted_avg = np.sum(good_bands['avg_thickness'] * good_bands['weight']) / total_weight
    print(f"加权平均厚度：{weighted_avg:.4f}μm（基于{len(good_bands)}个区间）")
    return weighted_avg, good_bands


# -------------------------- 7. 可视化与报告函数：增加空值防护 --------------------------
def plot_spectrum_with_extrema(data_valid, extrema, title, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(data_valid["lambda_nm"], data_valid["反射率 (%)"], color='grey', alpha=0.8, label='反射光谱')

    # 标记极值点（增加空值判断）
    if not extrema.empty:
        max_points = extrema[extrema['type'] == 'max']
        min_points = extrema[extrema['type'] == 'min']
        plt.scatter(max_points['lambda_nm'], max_points['反射率 (%)'], color=color1, s=50, label='极大值点', zorder=5)
        plt.scatter(min_points['lambda_nm'], min_points['反射率 (%)'], color=color2, s=50, label='极小值点', zorder=5)

    # 增加文字大小：标题从14改为16，坐标轴标签从12改为14
    plt.title(title, fontsize=16)
    plt.xlabel('波长 (nm)', fontsize=14)
    plt.ylabel('反射率 (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)  # 增加图例文字大小
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def save_text_report(models_results, final_thickness, incident_angle, file_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = f"output/txt/report_angle_{incident_angle}_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"薄膜厚度分析报告 - 入射角 {incident_angle}°\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析文件: {file_path}\n")
        f.write("=" * 60 + "\n\n")

        # Cauchy模型（增加空值判断）
        cauchy = models_results.get('cauchy', {})
        f.write("1. Cauchy 模型分析结果:\n")
        f.write(f"   - 加权平均厚度: {cauchy.get('final_thickness', 'N/A'):.4f} μm\n" if pd.notna(
            cauchy.get('final_thickness')) else "   - 加权平均厚度: N/A\n")
        details_c = cauchy.get('details')
        f.write(f"   - 有效分析区间数量: {len(details_c) if (details_c is not None and not details_c.empty) else 0}\n")

        if details_c is not None and not details_c.empty:
            overall_residual_std = details_c['residual_std'].mean()
            overall_residual_rmse = details_c['residual_rmse'].mean()
            f.write(f"   - 整体稳定性指标:\n")
            f.write(f"     残差标准差: {overall_residual_std:.4f}\n")
            f.write(f"     残差均方根误差: {overall_residual_rmse:.4f}\n")
            f.write("   - 各区间详情:\n")
            for _, row in details_c.iterrows():
                f.write(f"     波长范围: {row['start_lambda']:.0f}-{row['end_lambda']:.0f} nm, "
                        f"厚度: {row['avg_thickness']:.4f} μm, "
                        f"RSD: {row['rsd']:.2f}%, "
                        f"残差RMSE: {row['residual_rmse']:.4f}, "
                        f"权重: {row['norm_weight']:.2f}, "
                        f"折射率范围: {row['n_min']:.4f}-{row['n_max']:.4f}\n")

        # Sellmeier模型（同理）
        sellmeier = models_results.get('sellmeier', {})
        f.write("\n2. Sellmeier 模型分析结果:\n")
        f.write(f"   - 加权平均厚度: {sellmeier.get('final_thickness', 'N/A'):.4f} μm\n" if pd.notna(
            sellmeier.get('final_thickness')) else "   - 加权平均厚度: N/A\n")
        details_s = sellmeier.get('details')
        f.write(f"   - 有效分析区间数量: {len(details_s) if (details_s is not None and not details_s.empty) else 0}\n")

        if details_s is not None and not details_s.empty:
            overall_residual_std = details_s['residual_std'].mean()
            overall_residual_rmse = details_s['residual_rmse'].mean()
            f.write(f"   - 整体稳定性指标:\n")
            f.write(f"     残差标准差: {overall_residual_std:.4f}\n")
            f.write(f"     残差均方根误差: {overall_residual_rmse:.4f}\n")
            f.write("   - 各区间详情:\n")
            for _, row in details_s.iterrows():
                f.write(f"     波长范围: {row['start_lambda']:.0f}-{row['end_lambda']:.0f} nm, "
                        f"厚度: {row['avg_thickness']:.4f} μm, "
                        f"RSD: {row['rsd']:.2f}%, "
                        f"残差RMSE: {row['residual_rmse']:.4f}, "
                        f"权重: {row['norm_weight']:.2f}, "
                        f"折射率范围: {row['n_min']:.4f}-{row['n_max']:.4f}\n")

        # 综合结果
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"最终综合厚度: {final_thickness:.4f} μm\n" if pd.notna(final_thickness) else "最终综合厚度: N/A\n")
        f.write("=" * 40 + "\n")
    return txt_path


def generate_final_report(data_valid, models_results, final_thickness, incident_angle):
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(5, 2)

    # 子图1：反射光谱及极值点
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data_valid["lambda_nm"], data_valid["反射率 (%)"], color='grey', alpha=0.8, label='原始反射光谱')
    try:
        extrema = detect_interference_extrema(data_valid)
        if not extrema.empty:
            max_points = extrema[extrema['type'] == 'max']
            min_points = extrema[extrema['type'] == 'min']
            ax1.scatter(max_points['lambda_nm'], max_points['反射率 (%)'], color=color1, s=30, label='极大值点',
                        zorder=5)
            ax1.scatter(min_points['lambda_nm'], min_points['反射率 (%)'], color=color2, s=30, label='极小值点',
                        zorder=5)
    except Exception as e:
        print(f"绘制光谱图时极值点异常：{e}")

    # 标记优选区间（增加空值判断）
    colors = {'cauchy': color2, 'sellmeier': color1}
    for model, result in models_results.items():
        details = result.get('details')
        if details is not None and not details.empty:
            for _, row in details.iterrows():
                ax1.axvspan(row['start_lambda'], row['end_lambda'], color=colors[model], alpha=0.1)

    # 图例
    legend_elements = [
        plt.Line2D([0], [0], color='grey', lw=2, label='原始光谱'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color1, markersize=8, label='极大值点'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color2, markersize=8, label='极小值点'),
        Patch(facecolor=color2, alpha=0.3, label='Cauchy优选区间'),
        Patch(facecolor=color1, alpha=0.3, label='Sellmeier优选区间')
    ]
    ax1.legend(handles=legend_elements)
    ax1.set_title(f'入射角 {incident_angle}° 的反射光谱及优选分析区间', fontsize=14)
    ax1.set_xlabel('波长 (nm)')
    ax1.set_ylabel('反射率 (%)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 子图2-3：各模型区间权重（空值防护）
    for i, (model, result) in enumerate(models_results.items()):
        ax = fig.add_subplot(gs[1, i])
        details = result.get('details')
        if details is not None and not details.empty:
            details['band_label'] = details.apply(
                lambda row: f"{int(row['start_lambda'] / 1000)}-{int(row['end_lambda'] / 1000)}k", axis=1)
            bars = ax.bar(details['band_label'], details['norm_weight'], color=colors[model], alpha=0.7)
            ax.set_title(f'{model.capitalize()} 模型各区间归一化权重', fontsize=13)
            ax.set_ylabel('归一化权重')
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis='x', rotation=45)
            for bar, weight in zip(bars, details['norm_weight']):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{weight:.2f}', ha='center', va='bottom',
                        fontsize=9)
        else:
            ax.text(0.5, 0.5, '无有效区间', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model.capitalize()} 模型分析结果')
            ax.set_xticks([])

    # 子图4-5：厚度分布（空值防护）
    for i, (model, result) in enumerate(models_results.items()):
        ax = fig.add_subplot(gs[2, i])
        details = result.get('details')
        if details is not None and not details.empty:
            ax.hist(details['avg_thickness'], bins=5, alpha=0.7, color=colors[model])
            mean_thick = details['avg_thickness'].mean()
            ax.axvline(mean_thick, color='red', linestyle='--', linewidth=2, label=f'平均: {mean_thick:.4f} μm')
            ax.set_title(f'{model.capitalize()} 模型厚度分布', fontsize=13)
            ax.set_xlabel('厚度 (μm)')
            ax.set_ylabel('区间数量')
            ax.legend()
        else:
            ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])

    # 子图6-7：残差统计（空值防护）
    for i, (model, result) in enumerate(models_results.items()):
        ax = fig.add_subplot(gs[3, i])
        details = result.get('details')
        if details is not None and not details.empty:
            stats_data = [
                details['residual_mean'].mean(),
                details['residual_std'].mean(),
                details['residual_rmse'].mean()
            ]
            bars = ax.bar(['平均残差', '残差标准差', '残差RMSE'], stats_data, color=color4, alpha=0.7)
            ax.set_title(f'{model.capitalize()} 模型残差统计', fontsize=13)
            ax.set_ylabel('值')
            ax.tick_params(axis='x', rotation=30)
            for bar, val in zip(bars, stats_data):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.4f}', ha='center', va='bottom',
                        fontsize=9)
        else:
            ax.text(0.5, 0.5, '无残差数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])

    # 子图8：最终结果汇总
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')
    t_c = models_results.get('cauchy', {}).get('final_thickness', 'N/A')
    t_s = models_results.get('sellmeier', {}).get('final_thickness', 'N/A')
    c_details = models_results.get('cauchy', {}).get('details')
    s_details = models_results.get('sellmeier', {}).get('details')
    c_count = len(c_details) if (c_details is not None and not c_details.empty) else 0
    s_count = len(s_details) if (s_details is not None and not s_details.empty) else 0
    c_rmse = c_details['residual_rmse'].mean() if (c_details is not None and not c_details.empty) else 'N/A'
    s_rmse = s_details['residual_rmse'].mean() if (s_details is not None and not s_details.empty) else 'N/A'

    summary_text = (
        f"最终分析结果汇总 (入射角: {incident_angle}°)\n\n"
        f"1. Cauchy 模型:\n"
        f"   - 加权平均厚度: {t_c:.4f} μm\n" if pd.notna(t_c) else "   - 加权平均厚度: N/A\n"
                                                                   f"   - 分析基于 {c_count} 个优良波长区间\n"
                                                                   f"   - 平均残差RMSE: {c_rmse:.4f}\n" if isinstance(
            c_rmse, (int, float)) else "   - 平均残差RMSE: N/A\n\n"
                                       f"2. Sellmeier 模型:\n"
                                       f"   - 加权平均厚度: {t_s:.4f} μm\n" if pd.notna(t_s) else "   - 加权平均厚度: N/A\n"
                                                                                                  f"   - 分析基于 {s_count} 个优良波长区间\n"
                                                                                                  f"   - 平均残差RMSE: {s_rmse:.4f}\n" if isinstance(
            s_rmse, (int, float)) else "   - 平均残差RMSE: N/A\n\n"
                                       f"----------------------------------------------------\n"
                                       f"综合两个模型的稳定性加权平均后，\n"
                                       f"最终计算得到的薄膜厚度为: {final_thickness:.4f} μm" if pd.notna(
            final_thickness) else "最终计算得到的薄膜厚度为: N/A"
    )
    ax8.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", fc='aliceblue', ec='steelblue', lw=1))

    plt.tight_layout()
    report_path = f"output/images/final_report_angle_{incident_angle}.png"
    plt.savefig(report_path, dpi=300)
    plt.close()
    return report_path


# -------------------------- 8. 主函数：增加日志输出，便于调试 --------------------------
def main(file_path, incident_angle):
    print("=" * 60)
    print(f"开始分析文件: {file_path}, 入射角: {incident_angle}°")
    print("=" * 60)
    try:
        # 步骤1：数据预处理
        data_valid = preprocess_data(file_path, incident_angle)
        print("步骤 1/6: 数据预处理完成。")

        # 步骤2：绘制光谱及极值点图
        try:
            extrema = detect_interference_extrema(data_valid)
            spec_path = f"output/images/反射光谱及干涉极值点 (入射角 {incident_angle}°).png"
            plot_spectrum_with_extrema(data_valid, extrema, f'反射光谱及干涉极值点（入射角 {incident_angle}°）',
                                       spec_path)
            print(f"已保存光谱图: {spec_path}")
        except Exception as e:
            print(f"生成光谱图失败: {e}")

        # 步骤3：模型参数（碳化硅适配）
        cauchy_params = {"A": 2.65, "B": 0.015, "C": 1e-7}  # Cauchy标准参数
        sellmeier_params = {"A1": 6.91, "B1": 0.202}  # Sellmeier碳化硅参数
        models_results = {}

        # 步骤4：各模型计算
        for model_name, params in [("cauchy", cauchy_params), ("sellmeier", sellmeier_params)]:
            print(f"\n--- 正在为 [{model_name.capitalize()}] 模型寻找最佳分析区间... ---")
            bands_df = find_best_bands(data_valid, incident_angle, model_name, params)

            if bands_df.empty:
                print(f"警告: {model_name.capitalize()} 模型未找到有效区间，跳过加权计算")
                models_results[model_name] = {'final_thickness': np.nan, 'details': None}
                continue

            # 计算加权厚度
            weighted_thickness, good_bands = get_weighted_thickness(bands_df)
            if pd.isna(weighted_thickness):
                print(f"警告: {model_name.capitalize()} 模型无法计算加权厚度")
                models_results[model_name] = {'final_thickness': np.nan, 'details': None}
            else:
                models_results[model_name] = {'final_thickness': weighted_thickness, 'details': good_bands}
                print(f"步骤 3/6: {model_name.capitalize()} 模型计算完成")

        # 步骤5：综合两个模型结果
        print("\n--- 正在综合两个模型的结果... ---")
        t_cauchy = models_results.get("cauchy", {}).get('final_thickness')
        t_sellmeier = models_results.get("sellmeier", {}).get('final_thickness')
        valid_thickness = [t for t in [t_cauchy, t_sellmeier] if pd.notna(t)]

        if not valid_thickness:
            final_thickness = np.nan
            print("错误: 两个模型均无有效厚度")
        else:
            # 基于残差RMSE计算模型权重
            c_details = models_results.get('cauchy', {}).get('details')
            s_details = models_results.get('sellmeier', {}).get('details')
            c_rmse = c_details['residual_rmse'].mean() if (c_details is not None and not c_details.empty) else np.inf
            s_rmse = s_details['residual_rmse'].mean() if (s_details is not None and not s_details.empty) else np.inf

            # 权重 = 1/残差RMSE（残差越小，权重越大）
            weight_c = 1 / c_rmse if c_rmse != np.inf else 0
            weight_s = 1 / s_rmse if s_rmse != np.inf else 0
            total_weight = weight_c + weight_s

            if total_weight == 0:
                final_thickness = np.mean(valid_thickness)
                print(f"模型权重计算失败，使用简单平均: {final_thickness:.4f}μm")
            else:
                norm_c = weight_c / total_weight
                norm_s = weight_s / total_weight
                final_thickness = (t_cauchy * norm_c) + (t_sellmeier * norm_s)
                print(f"最终综合厚度: {final_thickness:.4f} μm")
                print(f"模型权重 - Cauchy: {norm_c:.2f}, Sellmeier: {norm_s:.2f}")

        # 步骤6：稳定性分析
        print("\n步骤 4/6: 执行稳定性分析...")
        # 简化稳定性分析（避免空值报错）
        stability_path = f"output/stability/model_consistency_{incident_angle}.png"
        try:
            plot_model_consistency(models_results, incident_angle, stability_path)
            print(f"稳定性分析图已保存: {stability_path}")
        except Exception as e:
            print(f"稳定性分析失败: {e}")

        # 步骤7：生成报告
        print("\n步骤 5/6: 生成可视化分析报告...")
        try:
            report_path = generate_final_report(data_valid, models_results, final_thickness, incident_angle)
            print(f"分析报告已保存至: {report_path}")
        except Exception as e:
            print(f"生成报告失败: {e}")

        # 步骤8：保存Excel和文本报告
        print("\n步骤 6/6: 保存结果数据...")
        try:
            excel_path = save_excel_results(models_results, final_thickness, incident_angle)
            print(f"Excel结果已保存至: {excel_path}")
        except Exception as e:
            print(f"保存Excel失败: {e}")

        try:
            txt_path = save_text_report(models_results, final_thickness, incident_angle, file_path)
            print(f"文本报告已保存至: {txt_path}")
        except Exception as e:
            print(f"保存文本报告失败: {e}")

        print("\n" + "=" * 60)
        print(f"分析完成！最终厚度: {final_thickness:.4f} μm" if pd.notna(final_thickness) else "分析完成，但无有效厚度")
        print("=" * 60)

    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")
        print("请检查文件路径和数据格式")


# -------------------------- 9. 其他辅助函数（保持原逻辑，增加空值防护） --------------------------
def plot_thickness_distribution(thickness_data, model_name, title, save_path=None):
    if len(thickness_data) == 0:
        print("无厚度数据，跳过绘图")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(thickness_data, bins=10, alpha=0.7, color=color3)
    plt.axvline(np.mean(thickness_data), color='red', linestyle='dashed', linewidth=2,
                label=f'平均值: {np.mean(thickness_data):.4f} μm')
    plt.title(f'{model_name} 模型厚度分布 - {title}', fontsize=13)
    plt.xlabel('厚度 (μm)')
    plt.ylabel('频率')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residual_analysis(residual_data, model_name, angle, save_path=None):
    if residual_data.empty:
        print("无残差数据，跳过绘图")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.scatter(residual_data["lambda_nm"], residual_data["residual"], color=color4, alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title(f'{model_name} 模型残差随波长变化')
    ax1.set_xlabel('波长 (nm)')
    ax1.set_ylabel('残差 (k_i - 预期k值)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.hist(residual_data["residual"], bins=10, alpha=0.7, color=color4)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_title(f'{model_name} 模型残差分布')
    ax2.set_xlabel('残差 (k_i - 预期k值)')
    ax2.set_ylabel('频率')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_qq_residuals(residual_data, model_name, angle, save_path=None):
    if residual_data.empty:
        print("无残差数据，跳过Q-Q图")
        return
    plt.figure(figsize=(8, 6))
    stats.probplot(residual_data["residual"], plot=plt)
    plt.title(f'{model_name} 模型残差Q-Q图 (入射角 {angle}°)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_consistency(models_results, angle, save_path=None):
    cauchy_data = models_results.get('cauchy', {}).get('details')
    sellmeier_data = models_results.get('sellmeier', {}).get('details')
    if cauchy_data is None or sellmeier_data is None or cauchy_data.empty or sellmeier_data.empty:
        print("无足够数据绘制模型一致性图")
        return

    plt.figure(figsize=(10, 8))
    plt.scatter(cauchy_data['avg_thickness'], sellmeier_data['avg_thickness'], color=color4, alpha=0.7, s=50)
    min_val = min(cauchy_data['avg_thickness'].min(), sellmeier_data['avg_thickness'].min())
    max_val = max(cauchy_data['avg_thickness'].max(), sellmeier_data['avg_thickness'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='理想一致性线 (y=x)')

    # 计算相关系数
    corr_coef = np.corrcoef(cauchy_data['avg_thickness'], sellmeier_data['avg_thickness'])[0, 1]
    plt.text(0.05, 0.95, f'相关系数: r = {corr_coef:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f'Cauchy与Sellmeier模型厚度结果一致性 (入射角 {angle}°)')
    plt.xlabel('Cauchy模型厚度 (μm)')
    plt.ylabel('Sellmeier模型厚度 (μm)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def save_excel_results(models_results, final_thickness, incident_angle):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"output/excel/results_angle_{incident_angle}_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        # 汇总结果
        cauchy_rmse = models_results.get('cauchy', {}).get('details', pd.DataFrame()).get('residual_rmse').mean() if \
            (models_results.get('cauchy', {}).get('details') is not None and not models_results.get('cauchy', {}).get(
                'details').empty) else 'N/A'
        sellmeier_rmse = models_results.get('sellmeier', {}).get('details', pd.DataFrame()).get(
            'residual_rmse').mean() if \
            (models_results.get('sellmeier', {}).get('details') is not None and not models_results.get('sellmeier',
                                                                                                       {}).get(
                'details').empty) else 'N/A'

        summary_data = {
            "模型": ["Cauchy", "Sellmeier", "综合结果"],
            "厚度 (μm)": [
                models_results.get('cauchy', {}).get('final_thickness', 'N/A'),
                models_results.get('sellmeier', {}).get('final_thickness', 'N/A'),
                final_thickness if pd.notna(final_thickness) else 'N/A'
            ],
            "有效区间数量": [
                len(models_results.get('cauchy', {}).get('details', [])) if (
                            models_results.get('cauchy', {}).get('details') is not None) else 0,
                len(models_results.get('sellmeier', {}).get('details', [])) if (
                            models_results.get('sellmeier', {}).get('details') is not None) else 0,
                "-"
            ],
            "平均残差RMSE": [
                cauchy_rmse if isinstance(cauchy_rmse, (int, float)) else 'N/A',
                sellmeier_rmse if isinstance(sellmeier_rmse, (int, float)) else 'N/A',
                "-"
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="汇总结果", index=False)

        # 各模型详细区间
        for model, result in models_results.items():
            details = result.get('details')
            if details is not None and not details.empty:
                details.to_excel(writer, sheet_name=f"{model}_区间详情", index=False)
    return excel_path


def perform_stability_analysis(models_results, data_valid, incident_angle):
    """简化稳定性分析，避免空值报错"""
    for model_name, result in models_results.items():
        details = result.get('details')
        if details is None or details.empty:
            print(f"{model_name}模型无有效数据，跳过稳定性分析")
            continue

        try:
            # 取第一个有效区间计算残差
            start_lambda = details.iloc[0]['start_lambda']
            end_lambda = details.iloc[0]['end_lambda']
            band_data = data_valid[(data_valid["lambda_nm"] >= start_lambda) & (data_valid["lambda_nm"] <= end_lambda)]
            extrema = detect_interference_extrema(band_data)
            model_params = {"A": 2.65, "B": 0.015, "C": 1e-7} if model_name == "cauchy" else {"A1": 6.91, "B1": 0.202}
            _, _, _, _, residual_data = calculate_thickness(extrema, incident_angle, model_name, **model_params)

            if residual_data is not None and not residual_data.empty:
                # 保存残差图
                res_plot_path = f"output/stability/{model_name}_residual_analysis_{incident_angle}.png"
                plot_residual_analysis(residual_data, model_name.capitalize(), incident_angle, res_plot_path)
                qq_plot_path = f"output/stability/{model_name}_residual_qq_{incident_angle}.png"
                plot_qq_residuals(residual_data, model_name.capitalize(), incident_angle, qq_plot_path)
                print(f"{model_name}模型残差图已保存")
        except Exception as e:
            print(f"{model_name}模型稳定性分析失败: {e}")

    # 模型一致性图
    consistency_path = f"output/stability/model_consistency_{incident_angle}.png"
    plot_model_consistency(models_results, incident_angle, consistency_path)


# -------------------------- 10. 运行入口（确保文件路径正确） --------------------------
if __name__ == "__main__":
    # 注意：请将文件路径改为你的实际路径（相对路径/绝对路径均可）
    file1_path = "附件1_processed.xlsx"
    angle1 = 10
    file2_path = "附件2_processed.xlsx"
    angle2 = 15

    # 运行分析
    print("=" * 80)
    print("开始分析附件1...")
    main(file_path=file1_path, incident_angle=angle1)

    print("\n" + "=" * 80)
    print("开始分析附件2...")
    main(file_path=file2_path, incident_angle=angle2)