# SiC多波束干涉厚度计算代码
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from datetime import datetime
from math import sin, sqrt, pi

# -------------------------- 1. 全局配置（依据标准与B题要求） --------------------------
SiC_REFRACTIVE_INDEX = 2.55
# 有效波长范围（标准测试范围3-200μm → 波数3333-50 cm⁻¹，取核心区间2500-5000nm即4000-2000 cm⁻¹）
VALID_WAVELENGTH_NM = (2500, 5000)
# 输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

# 绘图配置
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24

# -------------------------- 2. 数据预处理（依据B题附件数据格式） --------------------------
def preprocess_spectral_data(file_path, incident_angle):
    """
    处理B题附件的光谱数据（波数cm⁻¹ → 波长nm，筛选有效区间）
    :param file_path: 附件路径（附件1_processed.xlsx/附件2_processed.xlsx）
    :param incident_angle: 入射角（B题附件1为10°，附件2为15°）
    :return: 预处理后的数据（DataFrame）
    """
    # 读取数据（适配Excel格式）
    try:
        data = pd.read_excel(file_path, header=None, names=["波数(cm⁻¹)", "反射率(%)"])
    except Exception as e:
        raise ValueError(f"读取文件失败：{str(e)}")

    # 数据清洗：去除非数值、缺失值
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    # 波数→波长：λ(nm) = 1e7 / 波数(cm⁻¹)（B题原理推导基础）
    data["波长(nm)"] = 1e7 / data["波数(cm⁻¹)"]
    # 筛选有效波长区间
    data = data[(data["波长(nm)"] >= VALID_WAVELENGTH_NM[0]) & (data["波长(nm)"] <= VALID_WAVELENGTH_NM[1])]
    # 按波长降序排列（便于干涉级次计算）
    data = data.sort_values("波长(nm)", ascending=False).reset_index(drop=True)
    # 添加入射角信息
    data["入射角(°)"] = incident_angle

    # 数据量校验（B题多波束需足够点捕捉干涉条纹）
    if len(data) < 50:
        raise Warning(f"有效数据点仅{len(data)}个（建议≥50个），可能影响多波束极值检测")

    return data


# -------------------------- 3. 多波束干涉极值点检测（B题图2多波束特征） --------------------------
def detect_multibeam_extrema(data):
    """
    检测多波束干涉的极大值/极小值点（适配B题多波束尖锐峰谷特征）
    :param data: 预处理后的光谱数据
    :return: 极值点数据（DataFrame）、极大值索引、极小值索引
    """
    reflectance = data["反射率(%)"].values
    wavelength = data["波长(nm)"].values

    # 多波束极值检测参数（峰谷更尖锐，降低高度阈值、提高显著性）
    # 极大值检测
    peak_indices, _ = find_peaks(
        reflectance,
        distance=5,  # 峰间距（多波束峰更密集）
        prominence=0.05,  # 峰显著性（多波束峰特征更明显）
        height=np.mean(reflectance) + 0.1 * np.std(reflectance)  # 峰高门槛
    )
    # 极小值检测（取负反射率的极大值）
    valley_indices, _ = find_peaks(
        -reflectance,
        distance=5,
        prominence=0.05,
        height=-(np.mean(reflectance) - 0.1 * np.std(reflectance))  # 谷深门槛
    )

    # 整合极值点
    extrema_data = pd.DataFrame({
        "波长(nm)": np.concatenate([wavelength[peak_indices], wavelength[valley_indices]]),
        "反射率(%)": np.concatenate([reflectance[peak_indices], reflectance[valley_indices]]),
        "极值类型": ["极大值"] * len(peak_indices) + ["极小值"] * len(valley_indices)
    }).sort_values("波长(nm)", ascending=False).reset_index(drop=True)

    # 极值点数量校验（B题多波束需至少3个极值点计算级次差）
    if len(extrema_data) < 3:
        raise ValueError(f"仅检测到{len(extrema_data)}个极值点，不足计算多波束干涉厚度")

    print(f"多波束极值检测结果：极大值{len(peak_indices)}个，极小值{len(valley_indices)}个，共{len(extrema_data)}个")
    return extrema_data, peak_indices, valley_indices


# -------------------------- 4. 多波束干涉厚度计算（基于GB/T 42905-2023公式） --------------------------
def calculate_multibeam_thickness(extrema_data, incident_angle):
    """
    依据GB/T 42905-2023公式（10.7、10.8）计算SiC外延层厚度
    :param extrema_data: 极值点数据
    :param incident_angle: 入射角（°）
    :return: 厚度计算结果（DataFrame）、平均厚度（μm）、厚度RSD（%）
    """
    n = SiC_REFRACTIVE_INDEX  # 标准给定SiC折射率
    angle_rad = incident_angle * pi / 180  # 入射角转弧度
    thickness_results = []

    # 步骤1：计算干涉级次基数k0（基于极值点波长差，B题多波束级次差为0.5）
    lambda_ref = extrema_data.iloc[0]["波长(nm)"]  # 参考波长（最长波长）
    k0_estimates = []

    for i in range(1, len(extrema_data)):
        lambda_i = extrema_data.iloc[i]["波长(nm)"]
        m_i = i * 0.5  # 峰-谷/谷-峰的级次差（多波束极值间隔为0.5级）
        delta_lambda = lambda_ref - lambda_i
        if delta_lambda < 10:  # 过滤微小波长差（避免计算误差）
            continue
        # 级次基数估算：k0 = (m_i * lambda_i) / delta_lambda
        k0 = (m_i * lambda_i) / delta_lambda
        k0_estimates.append(k0)

    if not k0_estimates:
        raise ValueError("无法估算干涉级次基数，需更多有效极值点")
    k0 = np.median(k0_estimates)  # 用中位数抗异常值（多波束稳定性更优）

    # 步骤2：计算每个极值点对应的厚度
    for idx, row in extrema_data.iterrows():
        lambda_nm = row["波长(nm)"]
        lambda_μm = lambda_nm / 1000  # 转换为μm（标准厚度单位）
        m_i = idx * 0.5  # 当前极值点与参考点的级次差
        k_i = k0 + m_i  # 当前极值点的干涉级次

        # GB/T 42905-2023公式（10.8）：附加相移影响可忽略（小数点后第三位）
        # T = (k_i - 0.5) * λ_μm / (2 * sqrt(n² - sin²θ))
        denominator = 2 * sqrt(n ** 2 - sin(angle_rad) ** 2)
        thickness_μm = (k_i - 0.5) * lambda_μm / denominator

        # 筛选合理厚度（标准测试范围3-200μm）
        if 3 <= thickness_μm <= 200:
            thickness_results.append({
                "波长(nm)": lambda_nm,
                "反射率(%)": row["反射率(%)"],
                "极值类型": row["极值类型"],
                "干涉级次k_i": round(k_i, 4),
                "厚度(μm)": round(thickness_μm, 4)
            })

    # 结果整理
    result_df = pd.DataFrame(thickness_results)
    if result_df.empty:
        raise ValueError("无有效厚度计算结果，需调整极值点筛选条件")

    # 统计指标（符合标准精密度要求：单个实验室RSD≤1%）
    avg_thickness = result_df["厚度(μm)"].mean()
    std_thickness = result_df["厚度(μm)"].std()
    thickness_rsd = (std_thickness / avg_thickness) * 100 if avg_thickness != 0 else np.inf

    print(f"厚度统计：平均厚度={avg_thickness:.4f}μm，标准差={std_thickness:.4f}μm，RSD={thickness_rsd:.2f}%")
    return result_df, avg_thickness, thickness_rsd


# -------------------------- 5. 结果可视化（展示多波束光谱与极值点） --------------------------
def plot_multibeam_spectrum(data, extrema_data, peak_indices, valley_indices, incident_angle, save_path):
    """
    绘制多波束干涉光谱图（标注极值点，符合B题图2特征）
    :param data: 预处理后的数据
    :param extrema_data: 极值点数据
    :param peak_indices: 极大值索引
    :param valley_indices: 极小值索引
    :param incident_angle: 入射角（°）
    :param save_path: 图像保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制原始光谱
    ax.plot(data["波长(nm)"], data["反射率(%)"], color="black", alpha=0.8, linewidth=1.2, label="多波束反射光谱")
    # 标注极大值点
    ax.scatter(
        data.iloc[peak_indices]["波长(nm)"],
        data.iloc[peak_indices]["反射率(%)"],
        color="#00BA38", s=60, marker="^", label="多波束极大值点", zorder=5
    )
    # 标注极小值点
    ax.scatter(
        data.iloc[valley_indices]["波长(nm)"],
        data.iloc[valley_indices]["反射率(%)"],
        color="#F8766D", s=60, marker="v", label="多波束极小值点", zorder=5
    )

    # 图表配置
    ax.set_xlabel("波长 (nm)", fontsize=12)
    ax.set_ylabel("反射率 (%)", fontsize=12)
    ax.set_title(f"SiC多波束干涉光谱（入射角{incident_angle}°）", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")
    ax.legend(fontsize=10)
    ax.set_xlim(VALID_WAVELENGTH_NM[0], VALID_WAVELENGTH_NM[1])

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"光谱图已保存至：{save_path}")


# -------------------------- 6. 主函数（整合流程：数据→极值→厚度→输出） --------------------------
def main(file_path, incident_angle, file_label):
    """
    主流程：处理单个附件的多波束厚度计算
    :param file_path: 附件路径
    :param incident_angle: 入射角（°）
    :param file_label: 附件标签（如"附件1_10°"）
    :return: 最终平均厚度（μm）
    """
    print(f"\n{'=' * 60}\n开始处理{file_label}...\n{'=' * 60}")

    # 1. 数据预处理
    try:
        data = preprocess_spectral_data(file_path, incident_angle)
        print(
            f"数据预处理完成：有效波长范围{data['波长(nm)'].min():.0f}-{data['波长(nm)'].max():.0f}nm，共{len(data)}个数据点")
    except Exception as e:
        print(f"数据预处理失败：{str(e)}")
        return None

    # 2. 多波束极值检测
    try:
        extrema_data, peak_indices, valley_indices = detect_multibeam_extrema(data)
    except Exception as e:
        print(f"极值检测失败：{str(e)}")
        return None

    # 3. 厚度计算
    try:
        thickness_df, avg_thickness, thickness_rsd = calculate_multibeam_thickness(extrema_data, incident_angle)
    except Exception as e:
        print(f"厚度计算失败：{str(e)}")
        return None

    # 4. 结果可视化
    plot_path = f"{OUTPUT_DIR}/plots/{file_label}_多波束光谱图.png"
    plot_multibeam_spectrum(data, extrema_data, peak_indices, valley_indices, incident_angle, plot_path)

    # 5. 结果保存（Excel）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"{OUTPUT_DIR}/data/{file_label}_多波束厚度结果_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # 原始预处理数据
        data.to_excel(writer, sheet_name="预处理光谱数据", index=False)
        # 极值点数据
        extrema_data.to_excel(writer, sheet_name="多波束极值点", index=False)
        # 厚度计算结果
        thickness_df.to_excel(writer, sheet_name="厚度计算结果", index=False)
        # 统计汇总
        stats_df = pd.DataFrame({
            "统计项": ["平均厚度(μm)", "厚度标准差(μm)", "厚度RSD(%)", "入射角(°)", "有效极值点数"],
            "数值": [avg_thickness, thickness_df["厚度(μm)"].std(), thickness_rsd, incident_angle, len(extrema_data)]
        })
        stats_df.to_excel(writer, sheet_name="结果统计", index=False)

    print(f"结果数据已保存至：{excel_path}")
    print(f"{file_label}处理完成，最终平均厚度：{avg_thickness:.4f}μm\n")
    return avg_thickness


# -------------------------- 7. 执行入口（适配B题附件1、附件2） --------------------------
if __name__ == "__main__":
    # 配置B题附件路径与入射角（需根据实际文件位置调整）
    附件1路径 = "附件1_processed.xlsx"  # B题附件1：入射角10°
    附件2路径 = "附件2_processed.xlsx"  # B题附件2：入射角15°

    # 执行附件1处理
    附件1平均厚度 = main(
        file_path=附件1路径,
        incident_angle=10,
        file_label="附件1_入射角10°"
    )

    # 执行附件2处理
    附件2平均厚度 = main(
        file_path=附件2路径,
        incident_angle=15,
        file_label="附件2_入射角15°"
    )

    # 最终结果汇总
    print(f"\n{'=' * 80}")
    print("B题SiC多波束干涉厚度计算最终结果汇总")
    print(f"附件1（10°）平均厚度：{附件1平均厚度:.4f}μm" if 附件1平均厚度 else "附件1计算失败")
    print(f"附件2（15°）平均厚度：{附件2平均厚度:.4f}μm" if 附件2平均厚度 else "附件2计算失败")
    print(f"{'=' * 80}")