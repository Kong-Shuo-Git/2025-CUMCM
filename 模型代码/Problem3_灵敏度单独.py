import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter,detrend

# --- 1. 全局配置（与solution保持一致，确保精度） ---
warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24

# 颜色定义
COLOR_GROUP1 = '#FFD47D'  # 附件3数据
COLOR_GROUP2 = '#A5D497'  # 附件4数据
COLOR_MEAN = '#E76F51'  # 平均厚度标识

# 输出目录（确保与需求一致）
OUTPUT_DIR = 'output/灵敏度分析'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 核心参数（复用solution的准确配置）
MATERIALS_PARAMS = {
    'Si': {
        'n_fixed': 3.40,  # 固定硅折射率（与solution一致）
        'n0': 1.0003,  # 空气折射率
        'n2': 3.80,  # 衬底折射率
        'B': 0.08,  # 拟合B的参考值
        'C_fixed': 0.0003,  # 固定C参数
        'B_bounds': [0.075, 0.085],  # B的窄边界
        'phase_fixed': 0.0  # 固定相位偏移
    }
}

# 拟合控制参数（与solution保持一致）
MAX_FEV = 500000  # 足够迭代次数
SMOOTH_WINDOW = 17  # 与solution相同的平滑窗口
SMOOTH_ORDER = 2
MIN_PEAKS = 2
D_THRESHOLD = [3.0, 4.0]  # 硅外延层合理范围
PEAK_PROMINENCE = 0.01
WINDOW_SIZE = 30  # 自动区间选择窗口
VAR_THRESHOLD_RATIO = 1.0
VIRTUAL_PEAK_NUM = 4
FIT_TOL = 1e-15  # 高于机器精度


# --- 2. 物理模型（完全复用solution的多光束干涉模型） ---
def refractive_index_model(nu, B, mat_params):
    """固定n核心值，仅拟合B微调（与solution一致）"""
    nu_scaled = nu / 10000.0
    n = mat_params['n_fixed'] + B * (nu_scaled ** 2) + mat_params['C_fixed'] * (nu_scaled ** 4)
    return np.clip(n, 3.395, 3.405)  # 约束合理范围


def multi_beam_reflectivity(nu, d, B, offset, mat_params):
    """多光束干涉反射率模型（与solution完全一致）"""
    n0, n2 = mat_params['n0'], mat_params['n2']
    phase_shift = mat_params['phase_fixed']
    theta0 = np.deg2rad(multi_beam_reflectivity.theta_deg)

    # 波长与折射率计算
    lamda = 10000 / nu  # μm（统一单位）
    n1 = refractive_index_model(nu, B, mat_params)

    # 斯涅尔定律（避免定义域溢出）
    sin_theta1 = (n0 / n1) * np.sin(theta0)
    sin_theta1 = np.clip(sin_theta1, -0.999, 0.999)
    theta1 = np.arcsin(sin_theta1)

    # 光程差与相位差
    delta_L = 2 * d * np.cos(theta1)
    delta = (4 * np.pi / lamda) * delta_L + phase_shift

    # 动态反射系数
    r1 = (n0 * np.cos(theta0) - n1 * np.cos(theta1)) / (n0 * np.cos(theta0) + n1 * np.cos(theta1))
    r2 = (n1 * np.cos(theta1) - n2 * np.cos(theta1)) / (n1 * np.cos(theta1) + n2 * np.cos(theta1))

    # 艾里公式（反射率约束）
    R = (r1 ** 2 + r2 ** 2 + 2 * r1 * r2 * np.cos(delta)) / (1 + (r1 * r2) ** 2 + 2 * r1 * r2 * np.cos(delta))
    return np.clip(R * 100 + offset, 0, 100)


multi_beam_reflectivity.theta_deg = 0.0  # 静态入射角变量


# --- 3. 数据处理工具（复用solution的准确逻辑） ---
def auto_select_wavenumber_range(wavenumber, reflectivity):
    """自动选择有效干涉区间（与solution一致）"""
    reflectivity_detrend = detrend(reflectivity)
    # 滑动窗口方差
    window_var = np.convolve(
        np.square(reflectivity_detrend),
        np.ones(WINDOW_SIZE) / WINDOW_SIZE,
        mode='same'
    )
    var_mean = np.mean(window_var)
    var_threshold = var_mean * VAR_THRESHOLD_RATIO
    high_var_mask = window_var >= var_threshold

    if not np.any(high_var_mask):
        # 强制取硅干涉高发区400-800cm⁻¹
        mid_mask = (wavenumber >= 400) & (wavenumber <= 800)
        if np.sum(mid_mask) < 50:
            mid_start = int(len(wavenumber) * 0.2)
            mid_end = int(len(wavenumber) * 0.8)
            selected_mask = np.zeros_like(high_var_mask)
            selected_mask[mid_start:mid_end] = True
        else:
            selected_mask = mid_mask
    else:
        # 合并连续区间（优先400-800cm⁻¹）
        intervals = []
        start_idx = None
        for i, is_high in enumerate(high_var_mask):
            if is_high and start_idx is None:
                start_idx = i
            elif not is_high and start_idx is not None:
                interval_wave = (wavenumber[start_idx] + wavenumber[i - 1]) / 2
                if 400 <= interval_wave <= 800:
                    intervals.append((start_idx, i - 1))
                start_idx = None
        if start_idx is not None:
            interval_wave = (wavenumber[start_idx] + wavenumber[-1]) / 2
            if 400 <= interval_wave <= 800:
                intervals.append((start_idx, len(wavenumber) - 1))

        if not intervals:
            intervals = []
            start_idx = None
            for i, is_high in enumerate(high_var_mask):
                if is_high and start_idx is None:
                    start_idx = i
                elif not is_high and start_idx is not None:
                    intervals.append((start_idx, i - 1))
                    start_idx = None
            if start_idx is not None:
                intervals.append((start_idx, len(wavenumber) - 1))

        intervals.sort(key=lambda x: x[1] - x[0], reverse=True)
        best_start, best_end = intervals[0]
        best_start = max(0, best_start - WINDOW_SIZE // 2)
        best_end = min(len(wavenumber) - 1, best_end + WINDOW_SIZE // 2)
        selected_mask = np.zeros_like(high_var_mask)
        selected_mask[best_start:best_end] = True

    # 返回筛选后数据
    selected_data = pd.DataFrame({
        'wavenumber': wavenumber[selected_mask],
        'reflectivity': reflectivity[selected_mask]
    }).sort_values('wavenumber').reset_index(drop=True)
    return selected_data['wavenumber'].values, selected_data['reflectivity'].values


def generate_virtual_peaks(wavenumber, reflectivity, num_peaks):
    """生成虚拟峰（与solution一致）"""
    wave_min, wave_max = np.min(wavenumber), np.max(wavenumber)
    virtual_wave = np.linspace(wave_min + 15, wave_max - 15, num_peaks)
    virtual_peaks_idx = []
    for wave in virtual_wave:
        near_idx = np.argsort(np.abs(wavenumber - wave))[:8]
        max_idx = near_idx[np.argmax(reflectivity[near_idx])]
        virtual_peaks_idx.append(max_idx)
    virtual_peaks_idx = sorted(list(set(virtual_peaks_idx)))
    virtual_peaks_idx = [idx for idx in virtual_peaks_idx if 0 <= idx < len(wavenumber)]
    while len(virtual_peaks_idx) < 2:
        mid_idx = len(wavenumber) // 2
        virtual_peaks_idx.append(mid_idx)
        virtual_peaks_idx = sorted(list(set(virtual_peaks_idx)))
    return np.array(virtual_peaks_idx)


def preprocess_data(file_path):
    """数据加载与预处理（结合solution的波长转波数逻辑）"""
    try:
        if file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            data = pd.read_csv(file_path)
        # 处理波长转波数（与solution一致）
        if 'wavelength_μm' in data.columns:
            data['wavenumber'] = 10000 / data['wavelength_μm']
        else:
            data.columns = ['wavelength_μm', 'reflectivity']
            data['wavenumber'] = 10000 / data['wavelength_μm']

        data = data[(data['reflectivity'] > 0) & (data['reflectivity'] < 100)].dropna()
        data = data.sort_values('wavenumber').reset_index(drop=True)
        wavenumber_raw = data['wavenumber'].values
        reflectivity_raw = data['reflectivity'].values

        # 数据平滑（与solution参数一致）
        if len(reflectivity_raw) >= SMOOTH_WINDOW:
            reflectivity_smoothed = savgol_filter(reflectivity_raw, SMOOTH_WINDOW, SMOOTH_ORDER)
        else:
            return np.array([]), np.array([])

        # 自动选择有效区间（关键步骤，提升拟合准确性）
        wavenumber, reflectivity = auto_select_wavenumber_range(wavenumber_raw, reflectivity_smoothed)
        return wavenumber, reflectivity
    except Exception as e:
        print(f"数据预处理错误: {e}")
        return np.array([]), np.array([])


# --- 4. 拟合逻辑（复用solution的稳定拟合策略） ---
def get_initial_guess(wavenumber, reflectivity, theta_deg, mat_params):
    """初始值计算（与solution一致，避免边界问题）"""
    offset_guess = np.min(reflectivity)
    offset_guess = np.clip(offset_guess, 0, 50)
    B_ref = mat_params['B']

    # 峰值识别
    wave_range = np.max(wavenumber) - np.min(wavenumber)
    distance = max(3, int(wave_range / 18))
    peaks, _ = find_peaks(
        x=reflectivity,
        distance=distance,
        height=None,
        prominence=PEAK_PROMINENCE,
        width=[0.1, None],
        rel_height=0.5
    )

    # 峰数不足时生成虚拟峰
    if len(peaks) < MIN_PEAKS:
        peaks = generate_virtual_peaks(wavenumber, reflectivity, VIRTUAL_PEAK_NUM)

    # 初始厚度猜测（避免边界值，与solution一致）
    wave_mid = np.mean(wavenumber)
    n_approx = refractive_index_model(wave_mid, B_ref, mat_params)
    theta_rad = np.deg2rad(theta_deg)
    if len(peaks) >= 2:
        avg_delta_nu = np.mean(np.diff(wavenumber[peaks]))
        denominator = 2 * avg_delta_nu * np.sqrt(n_approx ** 2 - (mat_params['n0'] * np.sin(theta_rad)) ** 2)
        d_guess = 10000 / denominator if denominator != 0 else 3.7
    else:
        d_guess = 3.7  # 远离边界的初始值
    d_guess = np.clip(d_guess, D_THRESHOLD[0], D_THRESHOLD[1])

    return [d_guess, B_ref, offset_guess]


def get_param_bounds(mat_params):
    """参数边界（与solution一致，窄边界提升稳定性）"""
    lower = [
        D_THRESHOLD[0],  # d下限
        mat_params['B_bounds'][0],  # B下限
        0  # offset下限
    ]
    upper = [
        D_THRESHOLD[1],  # d上限
        mat_params['B_bounds'][1],  # B上限
        50  # offset上限
    ]
    return (lower, upper)


# --- 5. 灵敏度核心分析（确保与solution拟合逻辑一致） ---
def analyze_angle_sensitivity(file_path, theta_deg, material='Si'):
    """计算指定入射角下的厚度（复用solution的拟合流程）"""
    mat_params = MATERIALS_PARAMS.get(material, MATERIALS_PARAMS['Si'])
    wavenumber, reflectivity = preprocess_data(file_path)

    if len(wavenumber) < 50:  # 确保有足够数据点
        return np.nan

    # 初始值与边界
    p0 = get_initial_guess(wavenumber, reflectivity, theta_deg, mat_params)
    bounds = get_param_bounds(mat_params)

    try:
        multi_beam_reflectivity.theta_deg = theta_deg
        # 初始拟合（与solution一致的噪声权重）
        data_sigma = 0.01 * reflectivity + 0.05
        params, _ = curve_fit(
            f=lambda nu, d, B, offset: multi_beam_reflectivity(nu, d, B, offset, mat_params),
            xdata=wavenumber,
            ydata=reflectivity,
            p0=p0,
            bounds=bounds,
            sigma=data_sigma,
            absolute_sigma=False,
            maxfev=MAX_FEV,
            ftol=FIT_TOL,
            xtol=FIT_TOL,
            gtol=FIT_TOL,
            method='trf'
        )

        # 多轮迭代优化（与solution一致）
        for _ in range(2):
            fit_curve = multi_beam_reflectivity(wavenumber, *params, mat_params)
            residuals = reflectivity - fit_curve
            est_noise_std = np.std(residuals)
            data_sigma = np.abs(residuals) + est_noise_std * 0.03
            params, _ = curve_fit(
                f=lambda nu, d, B, offset: multi_beam_reflectivity(nu, d, B, offset, mat_params),
                xdata=wavenumber,
                ydata=reflectivity,
                p0=params,
                bounds=bounds,
                sigma=data_sigma,
                absolute_sigma=False,
                maxfev=MAX_FEV,
                method='trf'
            )
        return params[0]  # 返回厚度值
    except Exception as e:
        print(f"拟合失败（入射角{theta_deg}°）: {e}")
        return np.nan

# --- 6. 灵敏度图表（展示各角度厚度及平均值） ---
def plot_sensitivity_chart(results_dict):
    """绘制入射角-厚度关系图，包含平均厚度标注"""
    plt.figure(figsize=(14, 8))
    file_paths = list(results_dict.keys())
    if len(file_paths) < 2:
        return

    # 处理附件3数据
    angles1, thicknesses1 = results_dict[file_paths[0]]
    valid1 = ~np.isnan(thicknesses1)
    angles1_valid = np.array(angles1)[valid1]
    thicknesses1_valid = np.array(thicknesses1)[valid1]
    # 计算平均厚度
    mean1 = np.mean(thicknesses1_valid) if len(thicknesses1_valid) > 0 else np.nan

    # 处理附件4数据
    angles2, thicknesses2 = results_dict[file_paths[1]]
    valid2 = ~np.isnan(thicknesses2)
    angles2_valid = np.array(angles2)[valid2]
    thicknesses2_valid = np.array(thicknesses2)[valid2]
    # 计算平均厚度
    mean2 = np.mean(thicknesses2_valid) if len(thicknesses2_valid) > 0 else np.nan

    # 绘制各角度厚度
    bars1 = plt.bar(angles1_valid - 0.2, thicknesses1_valid, width=0.4,
            color=COLOR_GROUP1, label=os.path.basename(file_paths[0]))
    bars2 = plt.bar(angles2_valid + 0.2, thicknesses2_valid, width=0.4,
            color=COLOR_GROUP2, label=os.path.basename(file_paths[1]))

    # 为附件3数据添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}μm',
                 ha='center', va='bottom', fontsize=18)

    # 为附件4数据添加数值标签
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}μm',
                 ha='center', va='bottom', fontsize=18)

    # 标注平均厚度
    if not np.isnan(mean1):
        plt.axhline(y=mean1, color=COLOR_GROUP1, linestyle='--', linewidth=2,
                    label=f'{os.path.basename(file_paths[0])} 平均厚度: {mean1:.4f}μm')
    if not np.isnan(mean2):
        plt.axhline(y=mean2, color=COLOR_GROUP2, linestyle='--', linewidth=2,
                    label=f'{os.path.basename(file_paths[1])} 平均厚度: {mean2:.4f}μm')

    # 图表配置
    plt.xlabel('入射角 (°)', fontsize=24)
    plt.ylabel('拟合厚度 (μm)', fontsize=24)
    plt.title('不同入射角下半导体晶圆厚度拟合结果及灵敏度分析', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.legend(loc='best', fontsize=24)
    plt.xticks(range(8, 18))  # 入射角范围8-17°
    plt.tight_layout()

    # 保存图表
    save_path = os.path.join(OUTPUT_DIR, '入射角对半导体晶圆厚度拟合结果的灵敏度分析.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"灵敏度分析图表已保存至: {save_path}")


# --- 7. 主程序（按需求配置角度范围） ---
def main():
    # 配置文件与对应的入射角范围（附件3:8-12°，附件4:13-17°）
    files_config = {
        '附件3.xlsx': [8, 9, 10, 11, 12],  # 包含10°及新增角度
        '附件4.xlsx': [13, 14, 15, 16, 17]  # 包含15°及新增角度
    }

    results = {}
    for file_path, angles in files_config.items():
        if not os.path.exists(file_path):
            print(f"警告: 文件 '{file_path}' 不存在，跳过")
            continue
        # 计算每个入射角对应的厚度
        thicknesses = [analyze_angle_sensitivity(file_path, angle) for angle in angles]
        results[file_path] = (angles, thicknesses)

    # 生成包含平均厚度的灵敏度图表
    plot_sensitivity_chart(results)


if __name__ == "__main__":
    main()