import os
import scipy.io
import numpy as np

def convert_mat_to_txt(mat_file_path):
    # 检查输入文件是否为.mat文件
    if not mat_file_path.endswith('.mat'):
        raise ValueError("输入文件必须是.mat格式")
    
    # 检查文件是否存在
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"文件未找到: {mat_file_path}")
    
    # 读取.mat文件
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # 获取文件名
    file_name_without_ext = os.path.splitext(os.path.basename(mat_file_path))[0]
    
    # 创建.txt文件路径
    txt_file_path = f"{file_name_without_ext}.txt"
    
    # 将.mat数据写入.txt文件
    with open(txt_file_path, 'w') as txt_file:
        for key, value in mat_data.items():
            # 排除.mat文件中的一些默认键
            if key.startswith('__'):
                continue
            txt_file.write("{key}:")
            txt_file.write(np.array2string(value, separator=', '))
            txt_file.write("\n\n")
    
    print(f"转换成功，txt文件保存为: {txt_file_path}")

def convert_all_mat_to_txt_in_folder(folder_path="."):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹未找到: {folder_path}")
    
    # 遍历文件夹中的所有.mat文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            mat_file_path = os.path.join(folder_path, file_name)
            try:
                convert_mat_to_txt(mat_file_path)
            except Exception as e:
                print(f"转换文件 {file_name} 时出错: {str(e)}")

# 示例用法
convert_all_mat_to_txt_in_folder()