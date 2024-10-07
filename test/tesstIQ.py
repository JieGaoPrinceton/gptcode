import numpy as np
import matplotlib.pyplot as plt

# 生成示例 IQ 数据
t = np.arange(0, 1, 0.01)  # 时间向量
f = 5  # 信号频率
I = np.cos(2 * np.pi * f * t)  # 同相分量
Q = np.sin(2 * np.pi * f * t)  # 正交分量
IQ = I + 1j * Q  # 复数形式的 IQ 数据

# 显示 IQ 数据
print('IQ Data:')
print(IQ)

# 绘制 IQ 数据
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, I, label='In-phase (I)')
plt.title('In-phase Component (I)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, Q, label='Quadrature (Q)', color='orange')
plt.title('Quadrature Component (Q)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()