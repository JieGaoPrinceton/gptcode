import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby1, filtfilt, welch

# 模拟超声信号
np.random.seed(0)
sampling_rate = 1000  # 采样率，单位为 Hz
t = np.linspace(0, 1.0, sampling_rate)  # 1 秒的时间序列
freq1, freq2 = 50, 200  # 两个不同频率的信号
ultrasound_signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.3 * np.random.randn(len(t))

# 设计切比雪夫 I 型低通滤波器
cutoff_frequency = 100  # 截止频率，单位为 Hz
nyquist = 0.5 * sampling_rate
normal_cutoff = cutoff_frequency / nyquist
rp = 1  # 纹波，单位为 dB
b, a = cheby1(4, rp, normal_cutoff, btype='low', analog=False)

# 对超声信号进行低通滤波
filtered_signal = filtfilt(b, a, ultrasound_signal)

# 绘制原始信号和滤波后的信号
plt.figure(figsize=(12, 6))
plt.plot(t, ultrasound_signal, label='Original Signal', color='blue', alpha=0.5)
plt.plot(t, filtered_signal, label='Filtered Signal (Low-pass Chebyshev Type I)', color='red')
plt.title('Ultrasound Signal Filtering - Low-pass Chebyshev Type I Filter')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 使用 Welch 方法计算滤波前后的功率谱密度
f_original, Pxx_original = welch(ultrasound_signal, sampling_rate, nperseg=1024)
f_filtered, Pxx_filtered = welch(filtered_signal, sampling_rate, nperseg=1024)

# 绘制功率谱密度
plt.figure(figsize=(12, 6))
plt.semilogy(f_original, Pxx_original, label='Original Signal PSD', color='blue', alpha=0.5)
plt.semilogy(f_filtered, Pxx_filtered, label='Filtered Signal PSD (Low-pass Chebyshev Type I)', color='red')
plt.title('Power Spectral Density - Welch Method')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density [V**2/Hz]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()