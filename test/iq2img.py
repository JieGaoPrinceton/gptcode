import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from datetime import datetime
import os

def SVDfilter(IQ, cutoff):
    """
    SVD filter for an n-dimensional matrix.
    The temporal dimension has to be the last dimension.
    """
    initsize = IQ.shape
    
    # Adjust cutoff if necessary
    if cutoff[-1] > initsize[-1]:
        cutoff = np.arange(cutoff[0], initsize[-1] + 1)
    if len(cutoff) == 1:
        cutoff = np.arange(cutoff[0], initsize[-1] + 1)
    elif len(cutoff) == 2:
        cutoff = np.arange(cutoff[0], cutoff[1] + 1)
    
    # Special case handling
    if np.array_equal(cutoff, np.arange(1, initsize[-1] + 1)) or cutoff[0] < 2:
        return IQ
    
    # Reshape into Casorati matrix
    X = IQ.reshape(-1, initsize[-1])
    
    # Calculate SVD of the autocorrelated matrix
    U, _, _ = np.linalg.svd(np.dot(X.T, X))
    
    # Calculate the singular vectors
    V = np.dot(X, U)
    
    # Singular value decomposition
    Reconst = np.dot(V[:, cutoff - 1], U[:, cutoff - 1].T)
    
    # Reconstruction of the final filtered matrix
    IQf = Reconst.reshape(initsize)
    
    return IQf

# 生成示例 IQ 数据
t = np.linspace(0, 1, 100)  # 时间向量
f = 5  # 信号频率
I = np.cos(2 * np.pi * f * t)  # 同相分量
Q = np.sin(2 * np.pi * f * t)  # 正交分量
IQ = I + 1j * Q  # 复数形式的 IQ 数据

# 将 IQ 数据扩展为 2D 矩阵
IQ = np.tile(IQ, (100, 1))

# 应用 SVD 滤波
cutoff = np.array([2, 5])
IQf = SVDfilter(IQ, cutoff)

# 包络检测：计算包络
# 取幅度作为实数输入
IQf_real = np.abs(IQf)
envelope = np.abs(hilbert(IQf_real, axis=-1))

# 绘制原始和滤波后的 IQ 数据
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(np.abs(IQ), aspect='auto', cmap='gray')
plt.title('Original IQ Data (Magnitude)')
plt.xlabel('Time')
plt.ylabel('Sample')

plt.subplot(2, 2, 2)
plt.imshow(np.angle(IQ), aspect='auto', cmap='gray')
plt.title('Original IQ Data (Phase)')
plt.xlabel('Time')
plt.ylabel('Sample')

plt.subplot(2, 2, 3)
plt.imshow(np.abs(IQf), aspect='auto', cmap='gray')
plt.title('Filtered IQ Data (Magnitude)')
plt.xlabel('Time')
plt.ylabel('Sample')

plt.subplot(2, 2, 4)
plt.imshow(envelope, aspect='auto', cmap='gray')
plt.title('Envelope of Filtered IQ Data')
plt.xlabel('Time')
plt.ylabel('Sample')

plt.tight_layout()

# 生成时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 获取当前文件夹路径
current_folder = os.path.dirname(os.path.abspath(__file__))

# 生成文件名
filename = os.path.join(current_folder, f"iq_image_{timestamp}.png")

# 保存图形到文件
plt.savefig(filename)

# 显示图形
plt.show()

print(f"Image saved as {filename}")