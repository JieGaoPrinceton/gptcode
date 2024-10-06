import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, disk, closing
from skimage.measure import regionprops, label
from skimage.feature import canny
from scipy.ndimage import binary_fill_holes

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
    x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    ultrasound_image += np.exp(-((x - pos[0])**2 + (y - pos[1])**2) / (2 * 2**2))

# 添加一些噪声
ultrasound_image += 0.02 * np.random.rand(*image_size)

# 使用Canny边缘检测器来检测边缘
edges = canny(ultrasound_image, sigma=1.0)

# 填充边缘内部的区域
filled_edges = binary_fill_holes(edges)

# 应用Otsu阈值来二值化图像
thresh_val = threshold_otsu(ultrasound_image)
binary_image = ultrasound_image > thresh_val

# 使用形态学操作来清理图像
closed_image = closing(binary_image, disk(3))
cleaned_image = remove_small_objects(closed_image, min_size=20)

# 连通域分析来标记气泡
labeled_image, num_features = label(cleaned_image, return_num=True)
regions = regionprops(labeled_image)
coordinates = np.array([region.centroid for region in regions if 10 < region.area < 100])

# 绘制结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Ultrasound Image')
plt.imshow(ultrasound_image, cmap='gray')
plt.scatter(bubble_positions[:, 1], bubble_positions[:, 0], color='red', marker='x', label='True Bubble Positions')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Detected Bubble Positions')
plt.imshow(ultrasound_image, cmap='gray')
if len(coordinates) > 0:
    plt.scatter(coordinates[:, 1], coordinates[:, 0], color='blue', marker='o', facecolors='none', label='Detected Bubble Positions')
plt.legend()

plt.tight_layout()
plt.show()