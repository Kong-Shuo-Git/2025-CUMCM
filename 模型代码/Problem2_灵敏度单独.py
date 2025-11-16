import warnings
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from math import sin, sqrt, pi
import os
from datetime import datetime

# --- 1. 全局设置（沿用Solution，修复颜色bug）---
warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24
# 颜色配置（避免'midgray'错误，用标准颜色）
color_cauchy = "#FFD47D"  # Cauchy模型（和Solution一致）
color_sellmeier = "#A5D497"  # Sellmeier模型（和Solution一致）
color_avg = "#84ADDC"  # 平均值（和Solution一致）
grid_color = 'gray'  # 修复颜色错误
text_color = 'black'
bg_color = 'lightgray'

# 确保输出目录（用户要求的"灵敏度分析"文件夹）
output_root = "output/灵敏度分析"
os.makedirs(output_root, exist_ok=True)
os.makedirs(f"{output_root}/excel", exist_ok=True)
os.makedirs(f"{output_root}/images", exist_ok=True)


# --- 2. 核心物理模型（完全移植Solution，确保参数一致）---
def cauchy_refractive_index(lambda_μm, A=2.65, B=0.015, C=1e-7):
    """Cauchy模型：适配碳化硅红外波段（2.5-5μm）- 与Solution完全一致"""
    n = A + B / (lambda_μm ** 2) + C / (lambda_μm ** 4)
    return n


def sellmeier_refractive_index(lambda_μm, A1=6.91, B1=0.202):
    """Sellmeier模型：碳化硅标准参数 - 与Solution完全一致"""
    if lambda_μm ** 2 <= B1:
        raise ValueError(f"波长 {lambda_μm:.2f}μm 过小，需λ² > {B1}μm²")
    n_squared = 1 + (A1 * lambda_μm ** 2) / (lambda_μm ** 2 - B1)
    return sqrt(n_squared)


# --- 3. 数据预处理（完全移植Solution，确保数据范围一致）---
def preprocess_data(file_path, angle):
    try:
        data = pd.read_excel(file_path)
    except Exception:
        data = pd.read_csv(file_path)
    # 数据清洗与波长转换（和Solution逻辑一致）
    data.columns = ["波数 (cm-1)", "反射率 (%)"]
    data["波数 (cm-1)"] = pd.to_numeric(data["波数 (cm-1)"], errors='coerce')
    data["反射率 (%)"] = pd.to_numeric(data["反射率 (%)"], errors='coerce')
    data = data.dropna()
    data["lambda_nm"] = 1e7 / data["波数 (cm-1)"]
    # 修正波长范围（2500-5000nm，和Solution一致）
    valid_mask = (data["lambda_nm"] >= 2500) & (data["lambda_nm"] <= 5000)
    data_valid = data[valid_mask].sort_values("lambda_nm", ascending=False).reset_index(drop=True)
    if len(data_valid) < 50:
        raise ValueError(f"有效数据点仅{len(data_valid)}个，需≥50个")
    data_valid["incident_angle"] = angle
    print(
        f"预处理后[{os.path.basename(file_path)}]：波长{data_valid['lambda_nm'].min():.0f}-{data_valid['lambda_nm'].max():.0f}nm，共{len(data_valid)}个点")
    return data_valid


# --- 4. 极值点检测（完全移植Solution，确保检测精度）---
def detect_interference_extrema(data_valid):
    R = data_valid["反射率 (%)"].values
    lambda_nm = data_valid["lambda_nm"].values
    r_mean = np.mean(R)
    r_std = np.std(R)
    # 和Solution一致的检测参数
    min_height_max = r_mean + 0.2 * r_std
    min_height_min = r_mean - 0.8 * r_std
    min_distance = max(10, int(len(data_valid) * 0.02))
    prominence = 0.05
    # 检测峰谷
    peak_indices, _ = find_peaks(R, distance=min_distance, height=min_height_max, prominence=prominence)
    valley_indices, _ = find_peaks(-R, distance=min_distance, height=-min_height_min, prominence=prominence)
    # 整合极值点
    extrema = pd.concat([
        pd.DataFrame({"type": "max", "lambda_nm": lambda_nm[peak_indices], "反射率 (%)": R[peak_indices]}),
        pd.DataFrame({"type": "min", "lambda_nm": lambda_nm[valley_indices], "反射率 (%)": R[valley_indices]})
    ]).sort_values("lambda_nm", ascending=False).reset_index(drop=True)
    print(f"检测到极值点：共{len(extrema)}个（极大值{len(peak_indices)}个，极小值{len(valley_indices)}个）")
    if len(extrema) < 3:
        raise ValueError(f"极值点不足3个，需调整检测参数")
    return extrema


# --- 5. 厚度计算（完全移植Solution，含异常值过滤）---
def calculate_thickness(extrema, incident_angle, ref_model, **model_params):
    if len(extrema) < 2:
        return pd.DataFrame(), np.nan, np.nan, np.nan, pd.DataFrame()
    ref_index_func = cauchy_refractive_index if ref_model == "cauchy" else sellmeier_refractive_index
    angle_rad = incident_angle * pi / 180
    lambda0_nm = extrema.iloc[0]["lambda_nm"]
    k0_estimates = []
    # 和Solution一致的k0计算（波长差>10nm）
    for i in range(1, len(extrema)):
        m_i = i * 0.5
        lambda_i_nm = extrema.iloc[i]["lambda_nm"]
        lambda_diff = lambda0_nm - lambda_i_nm
        if lambda_diff > 10:
            k0_est = (m_i * lambda_i_nm) / lambda_diff
            k0_estimates.append(k0_est)
    if not k0_estimates:
        print("无有效k0估算值，需调整波长差阈值")
        return pd.DataFrame(), np.nan, np.nan, np.nan, pd.DataFrame()
    k0_final = np.median(k0_estimates)
    results = []
    residuals = []
    expected_k = []
    actual_k = []
    # 厚度计算与异常值过滤（0.1-100μm，和Solution一致）
    for i, row in extrema.iterrows():
        m_i = i * 0.5
        k_i = k0_final + m_i
        lambda_i_nm = row["lambda_nm"]
        lambda_i_μm = lambda_i_nm / 1000.0
        try:
            n_i = ref_index_func(lambda_i_μm, **model_params)
        except Exception as e:
            print(f"计算折射率失败：{e}，跳过该点")
            continue
        denominator = 2 * sqrt(n_i ** 2 - sin(angle_rad) ** 2)
        thickness = (k_i * lambda_i_μm) / denominator
        if 0.1 < thickness < 100:
            results.append({"lambda_nm": lambda_i_nm, "thickness_μm": thickness, "k_i": k_i, "n_i": n_i})
            expected_k_val = (2 * thickness * denominator) / lambda_i_μm
            residuals.append(k_i - expected_k_val)
            expected_k.append(expected_k_val)
            actual_k.append(k_i)
    thickness_df = pd.DataFrame(results)
    if thickness_df.empty:
        print("无有效厚度值，需调整厚度过滤范围")
        return thickness_df, np.nan, np.nan, np.nan, pd.DataFrame()
    # 统计量计算（和Solution一致）
    avg_thickness = thickness_df["thickness_μm"].mean()
    std_thickness = thickness_df["thickness_μm"].std()
    rsd = (std_thickness / avg_thickness) * 100 if avg_thickness != 0 else np.inf
    residual_data = pd.DataFrame({
        "lambda_nm": thickness_df["lambda_nm"], "actual_k": actual_k, "expected_k": expected_k,
        "residual": residuals, "thickness_μm": thickness_df["thickness_μm"]
    })
    print(f"{ref_model}模型：平均厚度{avg_thickness:.4f}μm，RSD={rsd:.2f}%")
    return thickness_df, avg_thickness, std_thickness, rsd, residual_data


# --- 6. 子区间搜索（完全移植Solution，确保区间数量一致）---
def find_best_bands(data_valid, incident_angle, model_name, model_params, window_size=800, step_size=200,
                    min_extrema=3):
    """和Solution完全一致的子区间搜索：window_size=800，step_size=200"""
    all_bands_results = []
    min_lambda = data_valid["lambda_nm"].min()
    max_lambda = data_valid["lambda_nm"].max()
    ref_index_func = cauchy_refractive_index if model_name == "cauchy" else sellmeier_refractive_index
    # 生成子区间（和Solution一致的范围）
    for start_lambda in np.arange(min_lambda, max_lambda - window_size + 100, step_size):
        end_lambda = start_lambda + window_size
        band_data = data_valid[(data_valid["lambda_nm"] >= start_lambda) & (data_valid["lambda_nm"] <= end_lambda)]
        if len(band_data) < 30:
            continue
        try:
            extrema = detect_interference_extrema(band_data)
            if len(extrema) < min_extrema:
                continue
            _, avg_thick, _, rsd, residuals = calculate_thickness(extrema, incident_angle, model_name, **model_params)
            if pd.notna(rsd) and rsd < 50 and residuals is not None and not residuals.empty:
                lambda_min_μm = start_lambda / 1000.0
                lambda_max_μm = end_lambda / 1000.0
                n_min = ref_index_func(lambda_max_μm, **model_params)
                n_max = ref_index_func(lambda_min_μm, **model_params)
                all_bands_results.append({
                    "start_lambda": start_lambda, "end_lambda": end_lambda, "avg_thickness": avg_thick, "rsd": rsd,
                    "extrema_count": len(extrema), "n_min": n_min, "n_max": n_max,
                    "residual_std": residuals["residual"].std(), "residual_mean": residuals["residual"].mean(),
                    "residual_rmse": np.sqrt(np.mean(residuals["residual"] ** 2))
                })
        except Exception as e:
            print(f"子区间[{start_lambda:.0f}-{end_lambda:.0f}nm]分析失败：{str(e)[:50]}")
            continue
    bands_df = pd.DataFrame(all_bands_results)
    print(f"{model_name}模型：找到{len(bands_df)}个有效子区间")
    return bands_df


# --- 7. 加权厚度计算（完全移植Solution，确保权重逻辑一致）---
def get_weighted_thickness(band_results_df, rsd_threshold=30.0, top_n=3):
    """和Solution一致的加权逻辑：RSD<30%，综合得分排序"""
    if band_results_df.empty:
        return np.nan, None
    # 筛选优良区间
    good_bands = band_results_df[
        (band_results_df['rsd'] < rsd_threshold) &
        (band_results_df['residual_rmse'] < 0.5)
        ].copy()
    # 无优良区间时取前3
    if good_bands.empty:
        band_results_df['combined_score'] = band_results_df['rsd'] * 0.6 + band_results_df['residual_rmse'] * 0.4
        good_bands = band_results_df.sort_values('combined_score').head(top_n).copy()
        print(f"无满足RSD<{rsd_threshold}%的区间，取综合得分前{top_n}的区间")
    # 计算权重（和Solution一致）
    good_bands['weight_rsd'] = 1 / (good_bands['rsd'] + 1e-6)
    good_bands['weight_residual'] = 1 / (good_bands['residual_rmse'] + 1e-6)
    good_bands['weight'] = (good_bands['weight_rsd'] + good_bands['weight_residual']) / 2
    total_weight = good_bands['weight'].sum()
    good_bands['norm_weight'] = good_bands['weight'] / total_weight
    # 加权平均
    weighted_avg = np.sum(good_bands['avg_thickness'] * good_bands['weight']) / total_weight
    print(f"加权平均厚度：{weighted_avg:.4f}μm（基于{len(good_bands)}个区间）")
    return weighted_avg, good_bands


# --- 8. 单角度分析（完全匹配Solution的单角度逻辑）---
def analyze_single_angle(file_path, incident_angle):
    """分析单个入射角，返回Cauchy/Sellmeier/平均厚度 - 和Solution逻辑一致"""
    print(f"\n{'=' * 50}\n开始分析：{os.path.basename(file_path)}，入射角{incident_angle}°")
    print('=' * 50)
    try:
        # 1. 数据预处理
        data_valid = preprocess_data(file_path, incident_angle)
        # 2. 模型参数（和Solution一致）
        cauchy_params = {"A": 2.65, "B": 0.015, "C": 1e-7}
        sellmeier_params = {"A1": 6.91, "B1": 0.202}
        models_results = {}
        # 3. 计算Cauchy模型
        print(f"\n--- Cauchy模型分析 ---")
        bands_cauchy = find_best_bands(data_valid, incident_angle, "cauchy", cauchy_params)
        if bands_cauchy.empty:
            print("Cauchy模型无有效区间")
            models_results['cauchy'] = np.nan
        else:
            cauchy_thick, _ = get_weighted_thickness(bands_cauchy)
            models_results['cauchy'] = cauchy_thick
        # 4. 计算Sellmeier模型
        print(f"\n--- Sellmeier模型分析 ---")
        bands_sellmeier = find_best_bands(data_valid, incident_angle, "sellmeier", sellmeier_params)
        if bands_sellmeier.empty:
            print("Sellmeier模型无有效区间")
            models_results['sellmeier'] = np.nan
        else:
            sellmeier_thick, _ = get_weighted_thickness(bands_sellmeier)
            models_results['sellmeier'] = sellmeier_thick
        # 5. 计算平均厚度
        valid_thicks = [v for v in models_results.values() if pd.notna(v)]
        avg_thick = np.mean(valid_thicks) if valid_thicks else np.nan
        models_results['avg'] = avg_thick
        # 打印结果
        print(f"\n{incident_angle}° 分析结果：")
        print(f"Cauchy模型厚度：{models_results['cauchy']:.4f}μm" if pd.notna(
            models_results['cauchy']) else "Cauchy模型厚度：N/A")
        print(f"Sellmeier模型厚度：{models_results['sellmeier']:.4f}μm" if pd.notna(
            models_results['sellmeier']) else "Sellmeier模型厚度：N/A")
        print(f"平均厚度：{avg_thick:.4f}μm" if pd.notna(avg_thick) else "平均厚度：N/A")
        return {
            "file": os.path.basename(file_path),
            "angle": incident_angle,
            "cauchy": models_results['cauchy'],
            "sellmeier": models_results['sellmeier'],
            "avg": avg_thick
        }
    except Exception as e:
        print(f"{incident_angle}° 分析失败：{str(e)}")
        return {
            "file": os.path.basename(file_path),
            "angle": incident_angle,
            "cauchy": np.nan,
            "sellmeier": np.nan,
            "avg": np.nan
        }


# --- 9. 汇总图片绘制（用户要求：所有角度平均厚度汇总）---
def plot_summary_thickness(all_results, save_path):
    """绘制附件1和附件2所有角度的厚度汇总图"""
    # 拆分附件1和附件2数据
    file1_results = [r for r in all_results if "附件1" in r["file"]]
    file2_results = [r for r in all_results if "附件2" in r["file"]]
    # 提取数据（过滤空值）
    # 附件1
    angles1 = [r["angle"] for r in file1_results if pd.notna(r["avg"])]
    cauchy1 = [r["cauchy"] for r in file1_results if pd.notna(r["avg"])]
    sellmeier1 = [r["sellmeier"] for r in file1_results if pd.notna(r["avg"])]
    avg1 = [r["avg"] for r in file1_results if pd.notna(r["avg"])]
    # 附件2
    angles2 = [r["angle"] for r in file2_results if pd.notna(r["avg"])]
    cauchy2 = [r["cauchy"] for r in file2_results if pd.notna(r["avg"])]
    sellmeier2 = [r["sellmeier"] for r in file2_results if pd.notna(r["avg"])]
    avg2 = [r["avg"] for r in file2_results if pd.notna(r["avg"])]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    width = 0.25  # 条形图宽度
    x1 = np.arange(len(angles1))
    x2 = np.arange(len(angles2))

    # 附件1子图
    ax1.bar(x1 - width, cauchy1, width, color=color_cauchy, alpha=0.8, label="Cauchy模型")
    ax1.bar(x1, sellmeier1, width, color=color_sellmeier, alpha=0.8, label="Sellmeier模型")
    ax1.bar(x1 + width, avg1, width, color=color_avg, alpha=0.8, label="平均厚度")
    ax1.set_title("附件1_processed.xlsx 厚度结果（入射角8-12°）", fontsize=14)
    ax1.set_xlabel("入射角 (°)", fontsize=12)
    ax1.set_ylabel("厚度 (μm)", fontsize=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(angles1)
    ax1.grid(True, linestyle='--', alpha=0.5, color=grid_color)
    ax1.legend()
    # 标注数值
    for x, y in zip(x1, avg1):
        ax1.text(x + width, y + 0.02, f"{y:.4f}", ha='center', va='bottom', fontsize=9)

    # 附件2子图
    ax2.bar(x2 - width, cauchy2, width, color=color_cauchy, alpha=0.8, label="Cauchy模型")
    ax2.bar(x2, sellmeier2, width, color=color_sellmeier, alpha=0.8, label="Sellmeier模型")
    ax2.bar(x2 + width, avg2, width, color=color_avg, alpha=0.8, label="平均厚度")
    ax2.set_title("附件2_processed.xlsx 厚度结果（入射角13-17°）", fontsize=14)
    ax2.set_xlabel("入射角 (°)", fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(angles2)
    ax2.grid(True, linestyle='--', alpha=0.5, color=grid_color)
    ax2.legend()
    # 标注数值
    for x, y in zip(x2, avg2):
        ax2.text(x + width, y + 0.02, f"{y:.4f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n汇总图已保存：{save_path}")


# --- 10. 主函数（执行灵敏度分析，匹配用户角度要求）---
def angle_sensitivity_main(file1_path, file2_path):
    """
    灵敏度分析主流程：
    - 附件1：8°、9°、10°、11°、12°
    - 附件2：13°、14°、15°、16°、17°
    """
    # 1. 配置角度范围（用户要求）
    file1_angles = [8, 9, 10, 11, 12]  # 附件1：10°基础扩展
    file2_angles = [13, 14, 15, 16, 17]  # 附件2：15°基础扩展
    all_results = []

    # 2. 分析附件1所有角度
    print(f"{'=' * 60}\n开始附件1_processed.xlsx 灵敏度分析（角度：{file1_angles}°）")
    print('=' * 60)
    for angle in file1_angles:
        result = analyze_single_angle(file1_path, angle)
        all_results.append(result)

    # 3. 分析附件2所有角度
    print(f"\n{'=' * 60}\n开始附件2_processed.xlsx 灵敏度分析（角度：{file2_angles}°）")
    print('=' * 60)
    for angle in file2_angles:
        result = analyze_single_angle(file2_path, angle)
        all_results.append(result)

    # 4. 保存Excel结果
    results_df = pd.DataFrame(all_results)
    excel_path = f"{output_root}/excel/灵敏度分析结果.xlsx"
    results_df.to_excel(excel_path, index=False)
    print(f"\nExcel结果已保存：{excel_path}")

    # 5. 绘制汇总图片（用户要求：output/灵敏度分析）
    summary_plot_path = f"{output_root}/images/汇总厚度对比图.png"
    plot_summary_thickness(all_results, summary_plot_path)

    # 6. 输出统计信息
    print(f"\n{'=' * 60}")
    print("灵敏度分析完成！")
    print(f"结果文件路径：{output_root}")
    print(f"  - Excel结果：{excel_path}")
    print(f"  - 汇总图片：{summary_plot_path}")
    print('=' * 60)


# --- 11. 运行入口（用户仅需修改文件路径）---
if __name__ == "__main__":
    # -------------------------- 用户配置区 --------------------------
    # 替换为你的附件1和附件2实际路径（绝对路径如"D:/附件1_processed.xlsx"）
    FILE1_PATH = "附件1_processed.xlsx"
    FILE2_PATH = "附件2_processed.xlsx"
    # -------------------------- 执行分析 --------------------------
    angle_sensitivity_main(file1_path=FILE1_PATH, file2_path=FILE2_PATH)