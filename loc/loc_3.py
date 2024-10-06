import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter

##高斯滤波算法


# 模拟超声信号数据
np.random.seed(0)
image_size = (200, 200)
num_bubbles = 10

# 创建一个空白的图像
ultrasound_image = np.zeros(image_size)

# 随机生成气泡的位置
bubble_positions = np.random.randint(0, min(image_size), size=(num_bubbles, 2))

# 在图像中添加气泡信号
for pos in bubble_positions:
    ultrasound_image[pos[0], pos[1]] = 1

# 添加一些噪声
ultrasound_image += 0.1 * np.random.rand(*image_size)

# 应用高斯滤波来模拟超声图像的点扩散函数 (PSF)
filtered_image = gaussian_filter(ultrasound_image, sigma=2)

# 使用最大滤波器检测局部最大值
local_max = (filtered_image == maximum_filter(filtered_image, size=20)) & (filtered_image > 0.2)
coordinates = np.argwhere(local_max)

# 绘制结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Raw US Image')
plt.imshow(ultrasound_image, cmap='gray')
plt.scatter(bubble_positions[:, 1], bubble_positions[:, 0], color='red', marker='x', label='Real Bubble')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Microbubble Image')
plt.imshow(filtered_image, cmap='gray')
plt.scatter(coordinates[:, 1], coordinates[:, 0], color='blue', marker='o', facecolors='none', label='Detected Bubble')
plt.legend()

plt.tight_layout()
plt.show()