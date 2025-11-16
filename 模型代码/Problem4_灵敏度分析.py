# SiC多波束干涉厚度灵敏度分析代码（基于B题与红外反射法标准）
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from math import sin, sqrt, pi

# -------------------------- 1. 核心参数（源自B题与GB/T 42905-2023标准） --------------------------
SiC_REFRACTIVE_INDEX = 2.55  # 标准明确SiC折射率
VALID_WAVELENGTH_NM = (2500, 5000)  # 有效波长区间（对应标准3-200μm测试范围）
SENSITIVITY_ANGLE_GROUPS = {
    "附件1_processed.xlsx": [8, 9, 10, 11, 12],  # 附件1基础10°+新增角度
    "附件2_processed.xlsx": [13, 14, 15, 16, 17]  # 附件2基础15°+新增角度
}
OUTPUT_DIR = "output/灵敏度分析"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 自动创建灵敏度分析输出文件夹

# 绘图基础配置（确保图表清晰）
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150


# -------------------------- 2. 数据预处理（适配B题附件格式） --------------------------
def preprocess_data(file_path, incident_angle):
    """简化版数据处理：波数转波长+有效区间筛选+数据清洗"""
    # 读取附件数据（B题附件为Excel格式，无表头）
    data = pd.read_excel(file_path, header=None, names=["波数(cm⁻¹)", "反射率(%)"])
    # 清洗非数值/缺失值
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    # 波数→波长（B题原理：λ(nm)=1e7/波数(cm⁻¹)）
    data["波长(nm)"] = 1e7 / data["波数(cm⁻¹)"]
    # 筛选有效波长区间（排除干扰）
    data = data[(data["波长(nm)"] >= VALID_WAVELENGTH_NM[0]) & (data["波长(nm)"] <= VALID_WAVELENGTH_NM[1])]
    # 按波长降序（便于干涉级次计算）
    data = data.sort_values("波长(nm)", ascending=False).reset_index(drop=True)
    data["入射角(°)"] = incident_angle
    return data


# -------------------------- 3. 多波束极值检测（B题图2多波束特征适配） --------------------------
def detect_extrema(data):
    """简化版极值检测：仅保留多波束峰谷核心逻辑"""
    reflectance = data["反射率(%)"].values
    wavelength = data["波长(nm)"].values

    # 多波束峰检测（峰更尖锐，降低显著性阈值）
    peaks, _ = find_peaks(reflectance, distance=5, prominence=0.05,
                          height=np.mean(reflectance) + 0.1 * np.std(reflectance))
    # 多波束谷检测（取负反射率的峰）
    valleys, _ = find_peaks(-reflectance, distance=5, prominence=0.05,
                            height=-(np.mean(reflectance) - 0.1 * np.std(reflectance)))

    # 整合极值点（按波长排序）
    extrema = pd.DataFrame({
        "波长(nm)": np.concatenate([wavelength[peaks], wavelength[valleys]]),
        "反射率(%)": np.concatenate([reflectance[peaks], reflectance[valleys]]),
        "类型": ["峰"] * len(peaks) + ["谷"] * len(valleys)
    }).sort_values("波长(nm)", ascending=False).reset_index(drop=True)

    return extrema if len(extrema) >= 3 else None  # 多波束需至少3个极值点


# -------------------------- 4. 厚度计算（严格遵循GB/T 42905-2023公式） --------------------------
def calc_thickness(extrema, incident_angle):
    """简化版厚度计算：基于标准10.8公式（忽略微小相移影响）"""
    n = SiC_REFRACTIVE_INDEX
    angle_rad = incident_angle * pi / 180  # 入射角转弧度
    lambda_ref = extrema.iloc[0]["波长(nm)"]  # 参考波长（最长波长）
    k0_est = []

    # 估算干涉级次基数k0（多波束级次差为0.5）
    for i in range(1, len(extrema)):
        lambda_i = extrema.iloc[i]["波长(nm)"]
        delta_lambda = lambda_ref - lambda_i
        if delta_lambda < 10:
            continue
        k0 = (i * 0.5 * lambda_i) / delta_lambda  # 级次差m=i*0.5
        k0_est.append(k0)
    k0 = np.median(k0_est)  # 中位数抗异常值（多波束稳定性更优）

    # 计算每个极值点厚度并取平均
    thickness_list = []
    for idx, row in extrema.iterrows():
        lambda_μm = row["波长(nm)"] / 1000  # 转μm（标准厚度单位）
        k_i = k0 + idx * 0.5  # 当前级次
        # 标准公式：T=(k_i-0.5)*λ/(2*sqrt(n²-sin²θ))
        denom = 2 * sqrt(n ** 2 - sin(angle_rad) ** 2)
        thick = (k_i - 0.5) * lambda_μm / denom
        if 3 <= thick <= 200:  # 标准测试范围（3-200μm）
            thickness_list.append(thick)

    return np.mean(thickness_list) if thickness_list else None  # 返回平均厚度


# -------------------------- 5. 灵敏度分析核心逻辑（入射角→厚度响应） --------------------------
def sensitivity_analysis(file_path, angle_group):
    """针对单个附件的灵敏度分析：计算不同入射角的平均厚度"""
    thickness_results = []
    for angle in angle_group:
        # 流程：数据处理→极值检测→厚度计算
        data = preprocess_data(file_path, angle)
        extrema = detect_extrema(data)
        if extrema is None:
            continue
        avg_thick = calc_thickness(extrema, angle)
        if avg_thick is not None:
            thickness_results.append({"入射角(°)": angle, "平均厚度(μm)": round(avg_thick, 4)})
    return pd.DataFrame(thickness_results)


# -------------------------- 6. 结果可视化（汇总所有入射角-厚度数据） --------------------------
def plot_sensitivity_summary(results_dict):
    """绘制灵敏度分析汇总图：所有附件的入射角-厚度关系"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 区分附件1和附件2数据（用不同样式）
    colors = ["#2E86AB", "#A23B72"]
    markers = ["o", "s"]
    labels = ["附件1_processed（基础10°）", "附件2_processed（基础15°）"]

    for (file, df), color, marker, label in zip(results_dict.items(), colors, markers, labels):
        if not df.empty:
            ax.plot(
                df["入射角(°)"], df["平均厚度(μm)"],
                color=color, marker=marker, markersize=8, linewidth=2, label=label
            )
            # 标注数据点数值
            for _, row in df.iterrows():
                ax.text(
                    row["入射角(°)"], row["平均厚度(μm)"],
                    f'{row["平均厚度(μm)"]:.4f}',
                    ha="center", va="bottom", fontsize=9, color=color
                )

    # 图表配置（突出灵敏度分析主题）
    ax.set_xlabel("入射角（°）", fontsize=12, fontweight="bold")
    ax.set_ylabel("SiC外延层平均厚度（μm）", fontsize=12, fontweight="bold")
    ax.set_title("SiC多波束干涉厚度-入射角灵敏度分析", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_xlim(min(SENSITIVITY_ANGLE_GROUPS["附件1_processed.xlsx"]) - 0.5,
                max(SENSITIVITY_ANGLE_GROUPS["附件2_processed.xlsx"]) + 0.5)

    # 保存图表到灵敏度分析文件夹
    plot_path = os.path.join(OUTPUT_DIR, "SiC厚度-入射角灵敏度分析图.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"灵敏度分析图已保存至：{plot_path}")


# -------------------------- 7. 执行入口（一键运行灵敏度分析） --------------------------
if __name__ == "__main__":
    # 存储所有附件的灵敏度分析结果
    all_sensitivity_results = {}

    # 遍历两个附件，执行灵敏度分析
    for file_name, angle_group in SENSITIVITY_ANGLE_GROUPS.items():
        print(f"正在处理{file_name}的灵敏度分析（入射角：{angle_group}°）...")
        result_df = sensitivity_analysis(file_name, angle_group)
        all_sensitivity_results[file_name] = result_df
        print(f"{file_name}分析完成，有效数据点：{len(result_df)}个\n")

    # 绘制并保存汇总图
    plot_sensitivity_summary(all_sensitivity_results)
    print("灵敏度分析全部完成！结果图表位于：output/灵敏度分析")