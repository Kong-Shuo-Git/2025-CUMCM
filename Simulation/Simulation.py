import numpy as np
import matplotlib.pyplot as plt

# 修改原字体配置部分
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["mathtext.fontset"] = "stix"  # 更换为stix字体集，对数学符号支持更好
plt.rcParams["mathtext.rm"] = "SimHei"  # 数学文本的常规字体
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings("ignore")


class SiCThicknessMeasurementDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("碳化硅外延层红外多光束干涉测量系统")
        self.root.geometry("1400x900")
        # 默认参数
        self.incident_angle = 10  # 入射角(度)
        self.thickness = 7.32  # 外延层厚度(μm)
        self.n_air = 1.0  # 空气折射率
        self.n_sic = 2.52  # 碳化硅外延层折射率
        self.n_substrate = 3.05  # 衬底折射率
        self.wavelength_range = (2.0, 10.0)  # 波长范围(μm)
        self.num_beams = 5  # 显示的光束数量
        self.setup_ui()
        self.update_plots()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = ttk.LabelFrame(self.root, text="测量参数控制", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_control_panel(control_frame)
        self.setup_plot_area(plot_frame)

    def setup_control_panel(self, parent):
        angle_frame = ttk.LabelFrame(parent, text="入射角设置")
        angle_frame.pack(fill=tk.X, pady=5)
        ttk.Label(angle_frame, text="入射角 θ (°):").pack(anchor=tk.W)
        self.angle_var = tk.DoubleVar(value=self.incident_angle)
        angle_scale = ttk.Scale(angle_frame, from_=5, to=30,
                                variable=self.angle_var, orient=tk.HORIZONTAL,
                                command=self.on_angle_change)
        angle_scale.pack(fill=tk.X, padx=5, pady=2)
        angle_value = ttk.Label(angle_frame, textvariable=self.angle_var)
        angle_value.pack()

        thickness_frame = ttk.LabelFrame(parent, text="外延层厚度设置")
        thickness_frame.pack(fill=tk.X, pady=5)
        ttk.Label(thickness_frame, text="厚度 d (μm):").pack(anchor=tk.W)
        self.thickness_var = tk.DoubleVar(value=self.thickness)
        thickness_scale = ttk.Scale(thickness_frame, from_=1, to=20,
                                    variable=self.thickness_var, orient=tk.HORIZONTAL,
                                    command=self.on_thickness_change)
        thickness_scale.pack(fill=tk.X, padx=5, pady=2)
        thickness_value = ttk.Label(thickness_frame, textvariable=self.thickness_var)
        thickness_value.pack()

        refractive_frame = ttk.LabelFrame(parent, text="材料折射率设置")
        refractive_frame.pack(fill=tk.X, pady=5)
        ttk.Label(refractive_frame, text="碳化硅折射率 n_SiC:").pack(anchor=tk.W)
        self.n_sic_var = tk.DoubleVar(value=self.n_sic)
        n_sic_scale = ttk.Scale(refractive_frame, from_=2.0, to=3.5,
                                variable=self.n_sic_var, orient=tk.HORIZONTAL,
                                command=self.on_refractive_change)
        n_sic_scale.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(refractive_frame, text="衬底折射率 n_sub:").pack(anchor=tk.W)
        self.n_sub_var = tk.DoubleVar(value=self.n_substrate)
        n_sub_scale = ttk.Scale(refractive_frame, from_=2.5, to=4.0,
                                variable=self.n_sub_var, orient=tk.HORIZONTAL,
                                command=self.on_refractive_change)
        n_sub_scale.pack(fill=tk.X, padx=5, pady=2)

        beam_frame = ttk.LabelFrame(parent, text="光束显示设置")
        beam_frame.pack(fill=tk.X, pady=5)
        ttk.Label(beam_frame, text="显示光束数量:").pack(anchor=tk.W)
        self.beam_var = tk.IntVar(value=self.num_beams)
        beam_scale = ttk.Scale(beam_frame, from_=1, to=10,
                               variable=self.beam_var, orient=tk.HORIZONTAL,
                               command=self.on_beam_change)
        beam_scale.pack(fill=tk.X, padx=5, pady=2)

        measure_btn = ttk.Button(parent, text="执行厚度测量",
                                 command=self.perform_measurement)
        measure_btn.pack(pady=10)

        result_frame = ttk.LabelFrame(parent, text="测量结果")
        result_frame.pack(fill=tk.X, pady=5)
        self.result_text = tk.Text(result_frame, height=8, width=35)
        self.result_text.pack(fill=tk.BOTH, padx=5, pady=5)

    def setup_plot_area(self, parent):
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax1 = self.fig.add_subplot(221)  # 光束传播示意图
        self.ax2 = self.fig.add_subplot(222)  # 干涉图案
        self.ax3 = self.fig.add_subplot(223)  # 反射率光谱
        self.ax4 = self.fig.add_subplot(224)  # 厚度计算结果

        self.fig.tight_layout(pad=3.0)

    def calculate_reflection_coefficient(self, n1, n2, angle_deg):
        """使用菲涅尔公式计算s波反射系数幅度（小角度近似可用）"""
        theta_i = np.radians(angle_deg)
        try:
            # 使用 Snell 定律求折射角
            sin_theta_t = (n1 / n2) * np.sin(theta_i)
            if abs(sin_theta_t) >= 1.0:
                return 1.0  # 全反射
            theta_t = np.arcsin(sin_theta_t)
            # s-偏振反射系数（更稳定用于多层膜）
            r = (n1 * np.cos(theta_i) - n2 * np.cos(theta_t)) / \
                (n1 * np.cos(theta_i) + n2 * np.cos(theta_t))
            return abs(r)
        except:
            return 1.0

    def calculate_phase_difference(self, wavelength, thickness, n, angle_deg):
        """计算单程往返的相位差 Δφ = (4π/λ) * n * d * cosθ_t"""
        theta_i = np.radians(angle_deg)
        try:
            sin_theta_t = (1.0 / n) * np.sin(theta_i)  # n_air = 1
            if abs(sin_theta_t) >= 1.0:
                return 0.0
            cos_theta_t = np.sqrt(1 - sin_theta_t ** 2)
            phase = 4 * np.pi * n * thickness * cos_theta_t / wavelength
            return phase
        except:
            return 0.0

    def airy_reflectance(self, R1, R2, phase):
        """Airy 公式：R = (R1 + R2 + 2√(R1R2)cosΔφ) / (1 + R1R2 + 2√(R1R2)cosΔφ)"""
        sqrt_R = np.sqrt(R1 * R2)
        numerator = R1 + R2 + 2 * sqrt_R * np.cos(phase)
        denominator = 1 + R1 * R2 + 2 * sqrt_R * np.cos(phase)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))

    def draw_beam_propagation(self):
        """绘制多光束干涉路径，优化显示逻辑"""
        self.ax1.clear()
        angle_rad = np.radians(self.incident_angle)
        thickness_val = self.thickness  # 局部变量，防止污染
        n_sic = self.n_sic
        num_beams = self.num_beams

        # Snell 定律计算折射角
        try:
            sin_theta2 = (self.n_air / n_sic) * np.sin(angle_rad)
            if abs(sin_theta2) >= 1.0:
                self.ax1.text(0, -thickness_val / 2, "全反射!", ha='center', va='center')
                self.ax1.set_xlim(-6, 6)
                self.ax1.set_ylim(-thickness_val - 1, 3)
                self.ax1.set_xlabel('位置 (μm)')
                self.ax1.set_ylabel('深度 (μm)')
                self.ax1.set_title('多光束干涉路径')
                self.ax1.grid(True, alpha=0.3)
                self.ax1.set_aspect('equal', adjustable='box')
                return
            theta2_rad = np.arcsin(sin_theta2)
        except Exception as e:
            self.ax1.text(0, -thickness_val / 2, f"角度计算异常: {str(e)}", ha='center', va='center')
            self.ax1.set_xlabel('位置 (μm)')
            self.ax1.set_ylabel('深度 (μm)')
            self.ax1.set_title('多光束干涉路径')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_aspect('equal', adjustable='box')
            return

        # 绘制界面
        self.ax1.axhline(y=0, color='black', linewidth=2, label='空气/SiC界面')
        self.ax1.axhline(y=-thickness_val, color='red', linewidth=2, label='SiC/衬底界面')

        # 入射光
        x_start, y_start = -5, 2
        x_end = 0
        self.ax1.plot([x_start, x_end], [y_start, 0], 'b-', lw=2, label='入射光')
        self.ax1.arrow(x_end - 1, 1, 0.8, -1, head_width=0.3, head_length=0.3, fc='b', ec='b')

        colors = plt.cm.viridis(np.linspace(0, 1, num_beams))
        max_x = 5  # 初始化最大x值

        for i in range(num_beams):
            dx = thickness_val * np.tan(theta2_rad)  # 单次斜边投影长度
            if i == 0:
                # 第一束：表面直接反射
                self.ax1.plot([0, -5], [0, 2], color=colors[i], lw=1.5, label=f'光束{i + 1}')
            else:
                points_x = [0]
                points_y = [0]
                for bounce in range(i):
                    # 向下走到底
                    x_next = points_x[-1] + dx
                    y_next = -thickness_val
                    points_x.append(x_next)
                    points_y.append(y_next)
                    # 向上返回表面
                    x_next += dx
                    y_next = 0
                    points_x.append(x_next)
                    points_y.append(y_next)
                # 最后一次向上出射
                x_final = points_x[-1] - 5
                y_final = 2
                points_x.append(x_final)
                points_y.append(y_final)
                self.ax1.plot(points_x, points_y, color=colors[i], lw=1.5, alpha=0.8, label=f'光束{i + 1}')

                # 更新最大x值
                current_max = max(points_x)
                if current_max > max_x:
                    max_x = current_max

        # 设置合适的显示范围
        self.ax1.set_xlim(-6, max_x + 1)
        self.ax1.set_ylim(-thickness_val - 1, 3)
        self.ax1.set_xlabel('位置 (μm)')
        self.ax1.set_ylabel('深度 (μm)')
        self.ax1.set_title('多光束干涉路径')

        # 优化图例显示
        handles, labels = self.ax1.get_legend_handles_labels()
        if len(handles) > 8:  # 限制图例数量，避免拥挤
            self.ax1.legend(handles[:4] + handles[-4:], labels[:4] + labels[-4:],
                            fontsize=7, loc='upper right', ncol=2)
        else:
            self.ax1.legend(fontsize=7, loc='upper right', ncol=2)

        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal', adjustable='box')

    def draw_interference_pattern(self):
        self.ax2.clear()
        wavelengths = np.linspace(2, 10, 500)
        R1 = self.calculate_reflection_coefficient(self.n_air, self.n_sic, self.incident_angle) ** 2
        R2 = self.calculate_reflection_coefficient(self.n_sic, self.n_substrate, self.incident_angle) ** 2
        phases = self.calculate_phase_difference(wavelengths, self.thickness, self.n_sic, self.incident_angle)
        reflectance = self.airy_reflectance(R1, R2, phases)

        self.ax2.plot(wavelengths, reflectance, 'b-', lw=2)
        self.ax2.set_xlabel('波长 (μm)')
        self.ax2.set_ylabel('反射率')
        self.ax2.set_title('干涉反射光谱')
        self.ax2.grid(True, alpha=0.3)

        # 自动找极大极小值（提高鲁棒性）
        maxima_idx = argrelextrema(reflectance, np.greater, order=10)[0]
        minima_idx = argrelextrema(reflectance, np.less, order=10)[0]

        if len(maxima_idx) > 0:
            self.ax2.plot(wavelengths[maxima_idx], reflectance[maxima_idx], 'ro', ms=4, label='极大值')
        if len(minima_idx) > 0:
            self.ax2.plot(wavelengths[minima_idx], reflectance[minima_idx], 'go', ms=4, label='极小值')

        self.ax2.legend()

    def draw_reflectance_spectrum(self):
        self.ax3.clear()
        wavenumbers = np.linspace(1000, 5000, 500)
        wavelengths = 1e4 / wavenumbers  # μm

        R1 = self.calculate_reflection_coefficient(self.n_air, self.n_sic, self.incident_angle) ** 2
        R2 = self.calculate_reflection_coefficient(self.n_sic, self.n_substrate, self.incident_angle) ** 2
        phases = self.calculate_phase_difference(wavelengths, self.thickness, self.n_sic, self.incident_angle)
        theoretical = self.airy_reflectance(R1, R2, phases)

        # 添加噪声模拟实验数据
        noise = np.random.normal(0, 0.015, theoretical.shape)
        experimental = np.clip(theoretical + noise, 0, 1)

        self.ax3.plot(wavenumbers, experimental, 'r-', alpha=0.7, label='实验数据')
        self.ax3.plot(wavenumbers, theoretical, 'b-', lw=2, label='理论拟合')
        self.ax3.axvspan(800, 1200, color='red', alpha=0.2, label='声子吸收区')

        self.ax3.set_xlabel(r'波数 (cm$^{-1}$)')
        self.ax3.set_ylabel('反射率')
        self.ax3.set_title('反射率光谱拟合')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)

    def calculate_thickness_from_interference(self):
        """基于相邻极大值间距估算厚度"""
        wavelengths = np.linspace(3, 8, 800)
        R1 = self.calculate_reflection_coefficient(self.n_air, self.n_sic, self.incident_angle) ** 2
        R2 = self.calculate_reflection_coefficient(self.n_sic, self.n_substrate, self.incident_angle) ** 2
        phases = self.calculate_phase_difference(wavelengths, self.thickness, self.n_sic, self.incident_angle)
        reflectance = self.airy_reflectance(R1, R2, phases)

        maxima_idx = argrelextrema(reflectance, np.greater, order=15)[0]

        # 修复：如果找不到足够的极值点，返回None而不是原始厚度
        if len(maxima_idx) < 2:
            return None

        # 取中间两相邻峰
        mid = len(maxima_idx) // 2
        idx1, idx2 = maxima_idx[mid], maxima_idx[mid + 1]
        lambda1, lambda2 = wavelengths[idx1], wavelengths[idx2]

        # 转换为波数差
        delta_sigma = abs(1 / lambda1 - 1 / lambda2) * 1e4  # cm⁻¹

        # 有效折射率修正入射角
        cos_term = np.sqrt(self.n_sic ** 2 - np.sin(np.radians(self.incident_angle)) ** 2)
        if cos_term == 0:  # 避免除以零
            return None

        calculated_d = 1 / (2 * delta_sigma * cos_term)  # 单位：cm → 转 μm
        return calculated_d * 1e4  # cm → μm

    def draw_thickness_results(self):
        self.ax4.clear()
        angles = np.arange(5, 31, 5)
        measured = []
        original_angle = self.incident_angle

        for ang in angles:
            self.incident_angle = ang
            thick = self.calculate_thickness_from_interference()
            # 添加轻微噪声模拟误差并处理可能的None值
            if thick is not None:
                thick *= (1 + np.random.normal(0, 0.015))
                measured.append(thick)

        self.incident_angle = original_angle

        # 确保有数据才绘图
        if measured:
            self.ax4.plot(angles[:len(measured)], measured, 'bo-', label='测量值', lw=2)
            self.ax4.axhline(y=self.thickness, color='r', ls='--', lw=2, label='真实值')

            avg = np.mean(measured)
            std = np.std(measured)
            err = abs(avg - self.thickness) / self.thickness * 100

            self.ax4.text(0.05, 0.95, f'平均: {avg:.3f} μm\n'
                                      f'偏差: ±{std:.3f} μm\n'
                                      f'误差: {err:.2f}%', transform=self.ax4.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.ax4.set_xlabel('入射角 (°)')
        self.ax4.set_ylabel('厚度 (μm)')
        self.ax4.set_title('多角度测量重复性')
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)

    def update_plots(self):
        try:
            self.draw_beam_propagation()
            self.draw_interference_pattern()
            self.draw_reflectance_spectrum()
            self.draw_thickness_results()
            self.canvas.draw()
        except Exception as e:
            # 显示错误信息而不是崩溃
            self.ax1.clear()
            self.ax1.text(0.5, 0.5, f"绘图错误: {str(e)}",
                          ha='center', va='center', transform=self.ax1.transAxes)
            self.canvas.draw()

    def on_angle_change(self, event):
        self.incident_angle = self.angle_var.get()
        self.update_plots()

    def on_thickness_change(self, event):
        self.thickness = self.thickness_var.get()
        self.update_plots()

    def on_refractive_change(self, event):
        self.n_sic = self.n_sic_var.get()
        self.n_substrate = self.n_sub_var.get()
        self.update_plots()

    def on_beam_change(self, event):
        self.num_beams = self.beam_var.get()
        self.update_plots()

    def perform_measurement(self):
        calc_thick = self.calculate_thickness_from_interference()
        R1 = self.calculate_reflection_coefficient(self.n_air, self.n_sic, self.incident_angle) ** 2
        R2 = self.calculate_reflection_coefficient(self.n_sic, self.n_substrate, self.incident_angle) ** 2

        # 处理测量失败的情况
        if calc_thick is None:
            result_str = "【测量失败】\n无法从干涉光谱中识别足够的极值点，请调整参数后重试。"
        else:
            result_str = f"""【厚度测量报告】
🔧 测量参数：
  入射角: {self.incident_angle}°
  SiC折射率: {self.n_sic:.3f}
  衬底折射率: {self.n_substrate:.3f}
📏 测量结果：
  计算厚度: {calc_thick:.3f} μm
  设定厚度: {self.thickness:.3f} μm
  相对误差: {abs(calc_thick - self.thickness) / self.thickness * 100:.2f}%
🔍 干涉条件评估：
  界面反射率 R1: {R1 * 100:.1f}%
  衬底反射率 R2: {R2 * 100:.1f}%
  是否满足强干涉: {'是' if R1 > 0.1 and R2 > 0.1 else '否'}
  分析光束数: {self.num_beams}
💡 方法说明：
  基于红外多光束干涉原理
  使用Airy公式建模反射谱
  通过波数域干涉周期反演厚度
"""

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_str)


if __name__ == "__main__":
    root = tk.Tk()
    app = SiCThicknessMeasurementDemo(root)
    root.mainloop()