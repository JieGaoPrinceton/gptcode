import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter, label
from skimage.measure import regionprops

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

# 模拟气泡的高斯形状
for pos in bubble_positions:
    x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    ultrasound_image += np.exp(-((x - pos[0])**2 + (y - pos[1])**2) / (2 * 2**2))

# 添加一些噪声
ultrasound_image += 0.02 * np.random.rand(*image_size)  # 进一步减少噪声强度

# 应用中值滤波来去除噪声
filtered_image = median_filter(ultrasound_image, size=3)

# 使用高斯滤波来平滑图像
smoothed_image = gaussian_filter(filtered_image, sigma=1)

# 使用自适应阈值来过滤噪声并突出气泡
threshold = np.mean(smoothed_image) + 1.5 * np.std(smoothed_image)  # 提高阈值以减少误检
binary_image = smoothed_image > threshold

# 连通域分析来标记气泡
labeled_image, num_features = label(binary_image)
regions = regionprops(labeled_image)
coordinates = np.array([region.centroid for region in regions if 10 < region.area < 100])  # 调整区域大小限制，进一步减少误检

# 绘制结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Ultrasound Image')
plt.imshow(ultrasound_image, cmap='gray')
plt.scatter(bubble_positions[:, 1], bubble_positions[:, 0], color='red', marker='x', label='True Bubble Positions')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Detected Bubble Positions')
plt.imshow(smoothed_image, cmap='gray')
if len(coordinates) > 0:
    plt.scatter(coordinates[:, 1], coordinates[:, 0], color='blue', marker='o', facecolors='none', label='Detected Bubble Positions')
plt.legend()

plt.tight_layout()
plt.show()