import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage.measure import label, regionprops
import imageio
from sklearn.cluster import KMeans

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
    frames.append(frame)

# 使用不同的滤波算法对每帧图像进行平滑处理
frames_gaussian = [gaussian_filter(frame, sigma=2) + 0.02 * np.random.rand(*image_size) for frame in frames]
frames_median = [median_filter(frame, size=3) + 0.02 * np.random.rand(*image_size) for frame in frames]
frames_uniform = [uniform_filter(frame, size=5) + 0.02 * np.random.rand(*image_size) for frame in frames]

# 合并滤波后的结果到一个GIF中
images = []
for idx, (frame_gaussian, frame_median, frame_uniform) in enumerate(zip(frames_gaussian, frames_median, frames_uniform)):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title(f'Frame {idx + 1} - Gaussian Filter')
    axes[0].imshow(frame_gaussian, cmap='gray')
    axes[1].set_title(f'Frame {idx + 1} - Median Filter')
    axes[1].imshow(frame_median, cmap='gray')
    axes[2].set_title(f'Frame {idx + 1} - Uniform Filter')
    axes[2].imshow(frame_uniform, cmap='gray')
    plt.tight_layout()
    fig.canvas.draw()

    # Convert the plot to an image array
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    images.append(image)
    plt.close(fig)

# 保存为GIF
imageio.mimsave('filter_comparison.gif', images, fps=2)

# 选择高斯滤波结果进行微泡跟踪
frames = frames_gaussian

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
imageio.mimsave('bubble_tracking3.gif', images, fps=2)