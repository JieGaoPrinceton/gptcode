import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter

# 模拟超声信号
np.random.seed(0)
sampling_rate = 1000  # 采样率，单位为 Hz
t = np.linspace(0, 1.0, sampling_rate)  # 1 秒的时间序列
freq1, freq2 = 50, 200  # 两个不同频率的信号
ultrasound_signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.3 * np.random.randn(len(t))

# 设计巴特沃斯低通滤波器
cutoff_frequency = 100  # 截止频率，单位为 Hz
nyquist = 0.5 * sampling_rate
normal_cutoff = cutoff_frequency / nyquist
b, a = butter(4, normal_cutoff, btype='low', analog=False)

# 对超声信号进行低通滤波
filtered_signal = filtfilt(b, a, ultrasound_signal)

# 绘制原始信号和滤波后的信号
plt.figure(figsize=(12, 6))
plt.plot(t, ultrasound_signal, label='Original Signal', color='blue', alpha=0.5)
plt.plot(t, filtered_signal, label='Filtered Signal (Low-pass)', color='red')
plt.title('Ultrasound Signal Filtering - Low-pass Butterworth Filter')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 设计巴特沃斯高通滤波器
cutoff_frequency_high = 150  # 高通截止频率，单位为 Hz
normal_cutoff_high = cutoff_frequency_high / nyquist
b_high, a_high = butter(4, normal_cutoff_high, btype='high', analog=False)

# 对超声信号进行高通滤波
filtered_signal_high = filtfilt(b_high, a_high, ultrasound_signal)

# 绘制高通滤波后的信号
plt.figure(figsize=(12, 6))
plt.plot(t, ultrasound_signal, label='Original Signal', color='blue', alpha=0.5)
plt.plot(t, filtered_signal_high, label='Filtered Signal (High-pass)', color='green')
plt.title('Ultrasound Signal Filtering - High-pass Butterworth Filter')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()