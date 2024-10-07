import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
import imageio
from sklearn.cluster import KMeans
from datetime import datetime

# 模拟多帧的超声信号数据用于微泡跟踪
np.random.seed(0)
num_frames = 20
image_size = (200, 200)
num_bubbles = 10
bubble_trajectories = [np.array([np.random.randint(0, image_size[0], size=2)]) for _ in range(num_bubbles)]

# 创建视频帧序列来模拟微泡的移动
frames = []
for frame_idx in range(num_frames):
    frame = np.zeros(image_size)
    new_positions = []
    for idx, bubble in enumerate(bubble_trajectories):
        # 模拟气泡的随机移动（最大移动距离为3像素）
        new_position = bubble[-1] + np.random.randint(-3, 4, size=2)
        new_position = np.clip(new_position, 0, image_size[0] - 1)  # 保持在边界内
        new_positions.append(new_position)
        bubble_trajectories[idx] = np.vstack([bubble, new_position])
        # 在帧中添加气泡信号
        x, y = new_position
        frame[x, y] = 1

    # 随机添加新的气泡
    if np.random.rand() < 0.1:  # 10%的概率添加新气泡
        new_bubble_position = np.random.randint(0, image_size[0], size=2)
        bubble_trajectories.append(np.array([new_bubble_position]))
        x, y = new_bubble_position
        frame[x, y] = 1
    frames.append(frame)

# 使用高斯滤波对每帧图像进行平滑处理，并生成带有噪声的图像
frames = [gaussian_filter(frame, sigma=2) + 0.02 * np.random.rand(*image_size) for frame in frames]

# 微泡跟踪算法（使用 KMeans 聚类）
tracked_positions = []
for frame_idx, frame in enumerate(frames):
    # 使用简单阈值对图像进行二值化
    binary_frame = frame > 0.1
    binary_frame = binary_frame.astype(np.uint8)

    # 寻找气泡的轮廓
    labeled_frame, num_features = label(binary_frame, return_num=True)
    regions = regionprops(labeled_frame)
    detected_positions = [region.centroid for region in regions]

    # 使用 KMeans 聚类进行微泡跟踪
    if frame_idx == 0:
        # 第一帧初始化
        tracked_positions = detected_positions
    else:
        if len(detected_positions) > 0:
            kmeans = KMeans(n_clusters=min(len(tracked_positions), len(detected_positions)), random_state=0)
            kmeans.fit(detected_positions)
            tracked_positions = kmeans.cluster_centers_

    # 绘制微泡的轨迹
    for idx, bubble in enumerate(bubble_trajectories):
        if len(bubble) > 1:
            bubble_path = np.array(bubble)
            plt.plot(bubble_path[:, 1], bubble_path[:, 0], 'r-', linewidth=1)

    # 保存追踪结果
    frames[frame_idx] = (frame, tracked_positions)

# 绘制跟踪结果并保存为GIF
images = []
for frame_idx, (frame, positions) in enumerate(frames):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f'Frame {frame_idx + 1} - Tracked Bubble Positions')
    ax.imshow(frame, cmap='gray')
    if len(positions) > 0:
        positions = np.array(positions)
        if positions.ndim == 2 and positions.shape[1] == 2:
            ax.scatter(positions[:, 1], positions[:, 0], color='blue', marker='o', facecolors='none', label='Tracked Bubble Positions')
    if len(ax.get_legend_handles_labels()[1]) > 0:
        ax.legend()
    fig.canvas.draw()

    # Convert the plot to an image array
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    images.append(image)
    plt.close(fig)

# 保存为GIF
# 获取当前时间的分和秒
current_time = datetime.now().strftime("%M_%S")

# 生成文件名
filename = f'tra_2_{current_time}.gif'

# 保存为GIF
imageio.mimsave(filename, images, fps=2)