import os
from scipy.io import loadmat
import numpy as np
from datetime import datetime

def read_and_print_mat_struct(filename):
    # 读取.mat文件
    mat_data = loadmat(filename)
    
    # 获取当前时间戳并创建输出.txt文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_file_path = os.path.join(os.path.dirname(filename), f"{timestamp}_output.txt")
    
    # 打开.txt文件写入数据
    with open(txt_file_path, 'w') as txt_file:
        # 打印并写入结构体内容
        for key, value in mat_data.items():
            if key.startswith('__'):
                continue
            txt_file.write("{key}:")
            print(f"{key}:")
            if isinstance(value, np.ndarray) or np.isscalar(value):
                value_str = np.array2string(value, separator=', ')
                txt_file.write(value_str + "\n\n")
                print(value_str)
            elif isinstance(value, (dict, list)):
                value_str = str(value)
                txt_file.write(value_str + "\n\n")
                print(value_str)
            else:
                value_str = str(value)
                txt_file.write(value_str + "\n\n")
                print(value_str)
            print("\n")
    
    print(f"数据已成功写入文件: {txt_file_path}")

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 示例用法
filename = os.path.join(current_dir, 'PALA_InVivoMouseTumor_001.mat')  # 替换为您的实际.mat文件名
read_and_print_mat_struct(filename)