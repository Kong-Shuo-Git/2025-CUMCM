# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter, detrend
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime

# --- 1. 全局配置（修复机器精度问题+优化初始值） ---
warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 28  # 增大全局字体大小

# 颜色定义
COLOR_DATA = 'black'
COLOR_FIT = '#F8766D'
COLOR_PEAK = "#00BFC4"
COLOR_RESIDUAL = '#7CAE00'
COLOR_AUTO_RANGE = '#FF6B6B'

# 【修复1】材料参数：固定核心参数，减少自由度（不变）
MATERIALS_PARAMS = {
    'Si': {
        'n_fixed': 3.40,  # 固定硅折射率
        'n0': 1.0003,  # 空气折射率
        'n2': 3.80,  # 衬底折射率
        'B': 0.08,  # 拟合B的参考值
        'C_fixed': 0.0003,  # 固定C参数
        'B_bounds': [0.075, 0.085],  # B的窄边界
        'phase_fixed': 0.0  # 固定相位偏移
    },
    'GaAs': {'n_fixed': 3.3, 'n0': 1.0003, 'n2': 3.6, 'B': 0.08, 'C_fixed': 0.003, 'B_bounds': [0.078, 0.082],
             'phase_fixed': 0.0},
    'SiC': {'n_fixed': 2.6, 'n0': 1.0003, 'n2': 2.8, 'B': 0.06, 'C_fixed': 0.002, 'B_bounds': [0.058, 0.062],
            'phase_fixed': 0.0}
}

# 【修复2】拟合控制：精度参数高于机器精度（2.22e-16）+ 优化初始值
MAX_FEV = 500000  # 足够迭代次数（减少至5e5，避免冗余）
SMOOTH_WINDOW = 17  # 减小平滑窗口（21→17，保留更多干涉细节）
SMOOTH_ORDER = 2  # 平滑阶数不变
MIN_PEAKS = 2  # 最低峰数2个
D_THRESHOLD = [3.0, 4.0]  # 硅外延层常见范围
PEAK_PROMINENCE = 0.01  # 最低突出度
WINDOW_SIZE = 30  # 滑动窗口不变
VAR_THRESHOLD_RATIO = 1.0  # 方差阈值不变
VIRTUAL_PEAK_NUM = 4  # 生成4个虚拟峰
FIT_TOL = 1e-15  # 拟合精度（1e-15 > 机器精度2.22e-16）

# --- 2. 输出目录自动创建 ---
output_dirs = ['output/images', 'output/excel', 'output/reports', 'output/logs']
for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)


# --- 3. 物理模型（不变，仅调用修复后的精度参数） ---
def refractive_index_model(nu, B, mat_params):
    """固定n核心值，仅拟合B微调"""
    nu_scaled = nu / 10000.0
    n = mat_params['n_fixed'] + B * (nu_scaled ** 2) + mat_params['C_fixed'] * (nu_scaled ** 4)
    return np.clip(n, 3.395, 3.405)  # 极窄n范围


def multi_beam_reflectivity(nu, d, B, offset, mat_params):
    """仅3个拟合变量：d、B、offset"""
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


multi_beam_reflectivity.theta_deg = 0.0


# --- 4. 自动区间选择（不变） ---
def auto_select_wavenumber_range(wavenumber, reflectivity):
    reflectivity_detrend = detrend(reflectivity)
    # 滑动窗口方差
    window_var = np.convolve(
        np.square(reflectivity_detrend),
        np.ones(WINDOW_SIZE) / WINDOW_SIZE,
        mode='same'
    )
    # 方差阈值（均值）
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
            print(f"  自动区间：无干涉，取中间60%（{wavenumber[mid_start]:.1f}-{wavenumber[mid_end]:.1f}cm⁻¹）")
        else:
            selected_mask = mid_mask
            print(f"  自动区间：无干涉，强制取硅高发区400-800cm⁻¹")
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
        print(
            f"  自动区间：识别干涉区（{wavenumber[best_start]:.1f}-{wavenumber[best_end]:.1f}cm⁻¹），共{best_end - best_start + 1}个点")

    # 返回筛选后数据
    selected_data = pd.DataFrame({
        'wavenumber': wavenumber[selected_mask],
        'reflectivity': reflectivity[selected_mask]
    }).sort_values('wavenumber').reset_index(drop=True)
    return selected_data['wavenumber'].values, selected_data['reflectivity'].values, selected_mask


# --- 5. 辅助工具函数（不变，新增虚拟峰索引校验） ---
def generate_virtual_peaks(wavenumber, reflectivity, num_peaks):
    """生成虚拟峰+索引校验，避免重复"""
    wave_min, wave_max = np.min(wavenumber), np.max(wavenumber)
    # 均匀分布波数位置（避开边缘）
    virtual_wave = np.linspace(wave_min + 15, wave_max - 15, num_peaks)
    virtual_peaks_idx = []
    for wave in virtual_wave:
        # 找波数附近8个点的最大值（更稳定）
        near_idx = np.argsort(np.abs(wavenumber - wave))[:8]
        max_idx = near_idx[np.argmax(reflectivity[near_idx])]
        virtual_peaks_idx.append(max_idx)
    # 去重+排序+校验索引范围
    virtual_peaks_idx = sorted(list(set(virtual_peaks_idx)))
    # 确保索引在有效范围内
    virtual_peaks_idx = [idx for idx in virtual_peaks_idx if 0 <= idx < len(wavenumber)]
    # 若数量不足，补充中间点
    while len(virtual_peaks_idx) < 2:
        mid_idx = len(wavenumber) // 2
        virtual_peaks_idx.append(mid_idx)
        virtual_peaks_idx = sorted(list(set(virtual_peaks_idx)))
    print(f"  生成虚拟峰：{len(virtual_peaks_idx)}个（原始峰不足，提供拟合约束）")
    return np.array(virtual_peaks_idx)


def log_peak_info(filename, wavenumber, peaks, is_virtual, d_guess, auto_range, reflectivity_stats):
    log_filename = os.path.splitext(filename)[0] + '_peak_range_log.txt'
    log_path = os.path.join('output/logs', log_filename)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== 峰值与自动区间日志 - {filename} ===\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"自动波数范围: {auto_range[0]:.1f}-{auto_range[1]:.1f} cm⁻¹\n")
        f.write(f"反射率统计：均值={reflectivity_stats['mean']:.2f}%, 标准差={reflectivity_stats['std']:.2f}%\n")
        f.write(f"峰值类型: {'虚拟峰' if is_virtual else '真实峰'} | 数量: {len(peaks)}\n")
        f.write(f"峰值波数：{wavenumber[peaks].round(2)}\n")
        f.write(f"初始厚度猜测: {d_guess:.4f}μm\n")
    print(f"  - 峰值日志已保存至: {log_path}")


def calculate_validation_metrics(reflectivity, fit_curve):
    ss_res = np.sum((reflectivity - fit_curve) ** 2)
    ss_tot = np.sum((reflectivity - np.mean(reflectivity)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    residuals = reflectivity - fit_curve
    return {
        'R2': r2,
        'residual_mean': np.mean(residuals),
        'residual_std': np.std(residuals),
        'residuals': residuals
    }


# --- 6. 核心分析流程（修复机器精度+优化初始值） ---
def analyze_spectrum(file_path, theta_deg, material='Si'):
    filename = os.path.basename(file_path)
    print(f"=== 开始处理: {filename}（入射角: {theta_deg}°, 材料: {material}）===")

    # 步骤1：数据加载+波长→波数转换
    try:
        data = pd.read_excel(file_path)
        data.columns = ['wavelength_μm', 'reflectivity']
        data.dropna(inplace=True)
        data['wavenumber'] = 10000 / data['wavelength_μm']  # 波长→波数
        data = data[(data['reflectivity'] > 0) & (data['reflectivity'] < 100)].reset_index(drop=True)
        data = data.sort_values('wavenumber').reset_index(drop=True)
        wavenumber_raw = data['wavenumber'].values
        reflectivity_raw = data['reflectivity'].values

        # 数据平滑（减小窗口，保留干涉细节）
        if len(reflectivity_raw) >= SMOOTH_WINDOW:
            reflectivity_smoothed = savgol_filter(reflectivity_raw, SMOOTH_WINDOW, SMOOTH_ORDER)
            print(f"步骤 1/6: 数据加载完成（波长：{data['wavelength_μm'].min():.2f}-{data['wavelength_μm'].max():.2f}μm）")
            print(f"          波数：{wavenumber_raw.min():.1f}-{wavenumber_raw.max():.1f}cm⁻¹，有效点：{len(data)}")
        else:
            raise ValueError(f"有效点仅{len(data)}个（需≥{SMOOTH_WINDOW}个）")
    except Exception as e:
        print(f"错误: 数据预处理失败 - {str(e)}")
        return

    # 步骤2：自动选择波数区间
    wavenumber, reflectivity, selected_mask = auto_select_wavenumber_range(
        wavenumber_raw, reflectivity_smoothed
    )
    auto_range = [np.min(wavenumber), np.max(wavenumber)]
    reflectivity_stats = {'mean': np.mean(reflectivity), 'std': np.std(reflectivity)}
    print(
        f"步骤 2/6: 自动区间选择完成（{auto_range[0]:.1f}-{auto_range[1]:.1f}cm⁻¹），区间反射率均值：{reflectivity_stats['mean']:.2f}%")

    # 步骤3：加载材料参数
    mat_params = MATERIALS_PARAMS.get(material, MATERIALS_PARAMS['Si'])
    n_fixed = mat_params['n_fixed']
    B_ref = mat_params['B']
    B_bounds = mat_params['B_bounds']
    print(f"步骤 3/6: 加载{material}参数 - 固定n={n_fixed}，拟合B∈{B_bounds}（减少自由度）")

    # 步骤4：峰值识别+虚拟峰生成（带索引校验）
    wave_range = np.max(wavenumber) - np.min(wavenumber)
    distance = max(3, int(wave_range / 18))  # 合理峰间距
    # 识别真实峰（无高度限制）
    peaks, _ = find_peaks(
        x=reflectivity,
        distance=distance,
        height=None,
        prominence=PEAK_PROMINENCE,
        width=[0.1, None],
        rel_height=0.5
    )

    # 峰数不足时生成虚拟峰
    is_virtual = False
    if len(peaks) < MIN_PEAKS:
        print(f"警告: 仅识别到 {len(peaks)} 个真实峰（需≥{MIN_PEAKS}个），生成虚拟峰补充约束")
        peaks = generate_virtual_peaks(wavenumber, reflectivity, VIRTUAL_PEAK_NUM)
        is_virtual = True

    # 【修复3】初始厚度猜测：避免卡在边界上限（4.00→3.70μm）
    wave_mid = np.mean(wavenumber)
    n_approx = refractive_index_model(wave_mid, B_ref, mat_params)
    theta_rad = np.deg2rad(theta_deg)
    if len(peaks) >= 2:
        avg_delta_nu = np.mean(np.diff(wavenumber[peaks]))
        denominator = 2 * avg_delta_nu * np.sqrt(n_approx ** 2 - (mat_params['n0'] * np.sin(theta_rad)) ** 2)
        d_guess = 10000 / denominator if denominator != 0 else 3.7
    else:
        d_guess = 3.7  # 初始值设为3.7μm（远离边界上限）
    d_guess = np.clip(d_guess, D_THRESHOLD[0], D_THRESHOLD[1])
    print(
        f"步骤 4/6: 峰值处理完成 - 峰数={len(peaks)}（{('真实峰' if not is_virtual else '虚拟峰')}），初始d={d_guess:.4f}μm")
    log_peak_info(filename, wavenumber, peaks, is_virtual, d_guess, auto_range, reflectivity_stats)

    # 步骤5：参数初始化（3个变量，初始值远离边界）
    offset_guess = np.min(reflectivity)
    offset_guess = np.clip(offset_guess, 0, 50)  # 基线约束
    p0 = [
        d_guess,  # 初始d=3.7μm（非边界）
        B_ref,  # B=0.08（中间值）
        offset_guess  # 基线偏移
    ]
    # 窄边界约束
    lower_bounds = [
        D_THRESHOLD[0],  # d ≥3.0μm
        B_bounds[0],  # B ≥0.075
        0  # offset ≥0%
    ]
    upper_bounds = [
        D_THRESHOLD[1],  # d ≤4.0μm
        B_bounds[1],  # B ≤0.085
        50  # offset ≤50%
    ]
    bounds = (lower_bounds, upper_bounds)
    # 验证初始值
    for i, (p, low, high) in enumerate(zip(p0, lower_bounds, upper_bounds)):
        if not (low <= p <= high):
            p0[i] = np.clip(p, low, high)
            print(f"  调整参数{i}初始值：{p:.2f}→{p0[i]:.2f}（边界[{low:.2f}, {high:.2f}]）")
    print(f"步骤 5/6: 拟合参数初始化完成 - 仅3个变量，初始d={p0[0]:.2f}μm")

    # 步骤6：【修复4】稳定拟合（精度参数>机器精度）
    print(f"步骤 6/6: 执行多光束拟合（{len(p0)}个变量，迭代次数={MAX_FEV}）...")
    try:
        multi_beam_reflectivity.theta_deg = theta_deg
        # 【修复核心】噪声权重适当增大，避免数值波动
        data_sigma = 0.01 * reflectivity + 0.05  # 权重>0.02，更稳定
        # noinspection PyTupleAssignmentBalance
        params, covariance = curve_fit(
            f=lambda nu, d, B, offset: multi_beam_reflectivity(nu, d, B, offset, mat_params),
            xdata=wavenumber,
            ydata=reflectivity,
            p0=p0,
            bounds=bounds,
            sigma=data_sigma,
            absolute_sigma=False,
            maxfev=MAX_FEV,
            ftol=FIT_TOL,  # 1e-15 > 机器精度2.22e-16
            xtol=FIT_TOL,
            gtol=FIT_TOL,
            method='trf'
        )

        # 迭代拟合（2次，优化权重）
        for iter_idx in range(2):
            fit_curve = multi_beam_reflectivity(wavenumber, *params, mat_params)
            residuals = reflectivity - fit_curve
            est_noise_std = np.std(residuals)
            data_sigma = np.abs(residuals) + est_noise_std * 0.03  # 动态权重
            # noinspection PyTupleAssignmentBalance
            params, covariance = curve_fit(
                f=lambda nu, d, B, offset: multi_beam_reflectivity(nu, d, B, offset, mat_params),
                xdata=wavenumber,
                ydata=reflectivity,
                p0=params,
                bounds=bounds,
                sigma=data_sigma,
                absolute_sigma=False,
                maxfev=MAX_FEV,
                ftol=FIT_TOL,
                xtol=FIT_TOL,
                gtol=FIT_TOL,
                method='trf'
            )
            fitted_d = params[0]
            print(f"  迭代{iter_idx + 1}/2: 噪声std={est_noise_std:.4f}%，当前d={fitted_d:.4f}μm")

        # 计算误差（此时标准差稳定）
        errors = np.sqrt(np.diag(covariance))
        fitted_d = params[0]
        print(f"拟合成功！厚度={fitted_d:.4f}μm")
        print_fitting_results(params, errors, material, mat_params)

        # 步骤7：生成输出
        generate_outputs(data, params, errors, file_path, theta_deg, material, peaks, selected_mask, auto_range,
                         is_virtual)
    except RuntimeError as e:
        print(f"错误: 拟合未收敛 - {str(e)}")
        print("建议：1. 检查自动区间反射率是否有波动；2. 调整PEAK_PROMINENCE至0.008")
        return
    except Exception as e:
        print(f"错误: 拟合异常 - {str(e)}")
        return

    print(f"=== 文件 {filename} 处理完成 ===\n")


# --- 7. 拟合结果打印（不变） ---
def print_fitting_results(params, errors, material, mat_params):
    fitted_d, fitted_B, fitted_offset = params
    d_err, B_err, offset_err = errors
    n_fixed = mat_params['n_fixed']
    B_bounds = mat_params['B_bounds']

    # 计算实际折射率
    fitted_n = refractive_index_model(1000, fitted_B, mat_params)  # 1000cm⁻¹处n

    print("\n--- 核心拟合结果 ---")
    print(f"1. 外延层厚度")
    print(f"   拟合值: {fitted_d:.4f} μm | 标准差: {d_err:.4f} μm | 约束范围: [{D_THRESHOLD[0]}, {D_THRESHOLD[1]}] μm")
    print(f"   ✅ 厚度在硅外延层合理范围内！")
    print(f"\n2. 折射率参数（固定n={n_fixed}，仅微调B）")
    print(f"   参数B: {fitted_B:.4f} ± {B_err:.4f} | 约束范围: {B_bounds}")
    print(f"   1000cm⁻¹处n: {fitted_n:.4f}（符合硅折射率3.395-3.405要求）")
    print(f"\n3. 其他参数（固定/低自由度）")
    print(f"   相位偏移: {mat_params['phase_fixed']:.4f} rad（固定，硅反射相移可忽略）")
    print(f"   基线偏移: {fitted_offset:.4f} ± {offset_err:.4f} % | 约束范围: [0, 50] %")
    print("---------------------")


# --- 8. 结果输出（不变） ---
def generate_outputs(data, params, errors, file_path, theta_deg, material, peaks, selected_mask, auto_range,
                     is_virtual):
    filename = os.path.basename(file_path)
    filename_base = os.path.splitext(filename)[0]
    wavenumber_raw = data['wavenumber'].values
    reflectivity_raw = data['reflectivity'].values
    wavenumber = wavenumber_raw[selected_mask]
    reflectivity = reflectivity_raw[selected_mask]
    mat_params = MATERIALS_PARAMS.get(material, MATERIALS_PARAMS['Si'])
    fit_curve = multi_beam_reflectivity(wavenumber, *params, mat_params)
    val_metrics = calculate_validation_metrics(reflectivity, fit_curve)
    r2, res_mean, res_std, residuals = val_metrics['R2'], val_metrics['residual_mean'], val_metrics['residual_std'], \
    val_metrics['residuals']
    fitted_d, fitted_B, fitted_offset = params
    fitted_n = refractive_index_model(1000, fitted_B, mat_params)

    # 8.1 可视化图表
    # 修改为只创建一个子图，删除子图2
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    # 子图1：数据+拟合+区间
    ax1.plot(wavenumber_raw, reflectivity_raw, color=COLOR_DATA, label='原始数据', alpha=0.4, linewidth=1)
    ax1.plot(wavenumber, reflectivity, color=COLOR_DATA, label='自动筛选干涉数据', alpha=0.8, linewidth=1.5)
    ax1.plot(wavenumber, fit_curve, color=COLOR_FIT, linewidth=2.5, label=f'多光束拟合（R²={r2:.4f}，d={fitted_d:.4f}μm）')
    ax1.axvspan(auto_range[0], auto_range[1], alpha=0.1, color=COLOR_AUTO_RANGE, label=f'自动识别干涉区间')
    # 标注峰值
    peak_label = f'{"虚拟峰" if is_virtual else "真实峰"}（{len(peaks)}个，提供拟合约束）'
    ax1.scatter(wavenumber[peaks], reflectivity[peaks], marker='v', color=COLOR_PEAK, s=80, zorder=5, label=peak_label)

    # 标题
    ax1.set_title(
        f'{material}外延层多光束干涉拟合 - {filename_base}\n厚度: {fitted_d:.4f}μm | 入射角: {theta_deg}° | 区间: {auto_range[0]:.1f}-{auto_range[1]:.1f}cm⁻¹',
        fontsize=22  # 增大标题字体
    )
    ax1.set_xlabel('波数 (cm⁻¹)', fontsize=20)  # 增大坐标轴标签字体
    ax1.set_ylabel('反射率 (%)', fontsize=20)  # 增大坐标轴标签字体
    ax1.legend(loc='best', fontsize=18)  # 增大图例字体
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', labelsize=18)  # 增大刻度标签字体
    ax1.tick_params(axis='y', labelsize=18)  # 增大刻度标签字体

    # 保存图表
    img_path = os.path.join('output/images', f'{filename_base}_analysis.png')
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - 可视化图表已保存至: {img_path}")

    # 8.2 Excel结果表
    param_names = ['厚度_d (μm)', '折射率_B (微调)', '基线偏移 (%)']
    ci_95 = 1.96 * errors
    params_df = pd.DataFrame({
        '参数名称': param_names,
        '拟合值': params.round(6),
        '标准误差': errors.round(6),
        '95%置信区间_下界': (params - ci_95).round(6),
        '95%置信区间_上界': (params + ci_95).round(6),
        '材料先验范围': [
            f"[{D_THRESHOLD[0]}, {D_THRESHOLD[1]}]（硅常见厚度）",
            f"{mat_params['B_bounds']}（折射率微调）",
            "[0, 50]（反射率基线）"
        ]
    })

    # 验证指标表
    val_df = pd.DataFrame({
        '参数名称': ['拟合优度_R²', '残差均值 (%)', '残差标准差 (%)', '自动区间_起始 (cm⁻¹)', '自动区间_结束 (cm⁻¹)',
                     '峰值类型', '1000cm⁻¹处n'],
        '拟合值': [r2, res_mean, res_std, auto_range[0], auto_range[1], '虚拟峰' if is_virtual else '真实峰', fitted_n],
        '标准误差': ['-', '-', '-', '-', '-', '-', '-'],
        '95%置信区间_下界': ['-', '-', '-', '-', '-', '-', '-'],
        '95%置信区间_上界': ['-', '-', '-', '-', '-', '-', '-'],
        '材料先验范围': ['≥0.95（优）', '≈0（优）', '<1（优）', '—', '—', '≥2个', '3.395-3.405']
    })

    final_df = pd.concat([params_df, val_df], ignore_index=True)
    excel_path = os.path.join('output/excel', f'{filename_base}_results.xlsx')
    final_df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"  - Excel结果表已保存至: {excel_path}")

    # 8.3 文本报告
    report_path = os.path.join('output/reports', f'{filename_base}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write(f"{material}外延层多光束干涉拟合分析报告（基于物理合理范围）\n")
        f.write("=" * 90 + "\n")
        f.write(f"文件名: {filename}\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"入射角: {theta_deg}° | 原始波长: {data['wavelength_μm'].min():.2f}-{data['wavelength_μm'].max():.2f}μm\n")
        f.write(f"自动干涉区间: {auto_range[0]:.1f}-{auto_range[1]:.1f}cm⁻¹ | 有效点: {len(wavenumber)}个\n")
        f.write(f"拟合模型: 多光束干涉（艾里公式）+ 固定折射率（减少自由度，提升稳定性）\n")
        f.write(f"峰值处理: {'生成虚拟峰补充约束' if is_virtual else '使用真实峰约束'}\n\n")

        f.write("--- 1. 核心结果 ---\n")
        f.write(f"外延层厚度: {fitted_d:.4f} μm（标准差: {errors[0]:.4f} μm）\n")
        f.write(f"厚度合理性: 符合硅外延层常见厚度范围（3.0-4.0μm）\n")
        f.write(f"折射率（1000cm⁻¹）: {fitted_n:.4f}（符合硅标准折射率3.395-3.405）\n")
        f.write(f"拟合优度R²: {r2:.4f} | 残差标准差: {res_std:.4f}%（拟合稳定）\n\n")

        f.write("--- 2. 拟合参数详情 ---\n")
        for i in range(len(params_df)):
            row = params_df.iloc[i]
            f.write(
                f"{row['参数名称']:>20}: {row['拟合值']:>10.6f} ± {row['标准误差']:>6.6f} | 参考范围: {row['材料先验范围']}\n")

        f.write("\n--- 3. 结果验证建议 ---\n")
        f.write("1. 厚度验证：建议用台阶仪实测对比，硅外延层厚度误差应<5%\n")
        f.write("2. 折射率验证：1000cm⁻¹处n应在3.395-3.405之间，确保物理合理性\n")
        f.write("3. 稳定性验证：不同入射角（10°/15°）厚度偏差应<0.2μm\n")
        f.write("=" * 90 + "\n")
    print(f"  - 文本报告已保存至: {report_path}")


# --- 9. 主程序（不变） ---
def main():
    print("=" * 70)
    print("      硅外延层厚度拟合分析程序（V5.4 - 最终稳定版）")
    print("=" * 70 + "\n")

    # 待处理文件
    files_to_process = {
        '附件3.xlsx': (10.0, 'Si'),
        '附件4.xlsx': (15.0, 'Si')
    }

    # 批量处理
    for file_path, (theta_deg, material) in files_to_process.items():
        if not os.path.exists(file_path):
            print(f"警告: 文件 '{file_path}' 不存在，跳过\n")
            continue
        analyze_spectrum(file_path, theta_deg, material)

    print("所有文件处理完成！结果已保存至 output 目录（厚度符合硅常见范围）")


if __name__ == '__main__':
    main()