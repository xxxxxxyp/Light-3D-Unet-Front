import os
import glob
import numpy as np
import SimpleITK as sitk
import pandas as pd

def analyze_dataset_v3(folder_path, suv_threshold=0.1):
    """
    suv_threshold: 建议设为 0.1 或 0.2，用于过滤空气背景噪声
    """
    stats = []
    file_list = glob.glob(os.path.join(folder_path, "*.nii.gz"))
    
    for f in file_list:
        img = sitk.ReadImage(f)
        array = sitk.GetArrayFromImage(img) # [Z, Y, X]
        
        # 1. 仅统计具有实际意义的 SUV 区域
        body_mask = array > suv_threshold
        
        # 2. 计算每层面积，过滤由于床板或手臂边缘产生的细碎噪声
        # 我们使用形态学开运算（可选）或直接寻找最大连通域
        slice_areas = body_mask.sum(axis=(1, 2))
        max_area_idx = np.argmax(slice_areas)
        max_area = slice_areas[max_area_idx]
        
        # 3. 过滤掉解剖结构不完整的切片（面积小于最大截面 60% 的都扔掉）
        valid_slices = np.where(slice_areas > max_area * 0.6)[0]
        
        widths = []
        for s in valid_slices:
            # 这一层的二值掩码
            layer = body_mask[s]
            # 找到包含人体的列索引
            cols = np.any(layer, axis=0)
            if np.any(cols):
                # 真实的像素跨度
                w = np.where(cols)[0][-1] - np.where(cols)[0][0]
                widths.append(w)
        
        if widths:
            # 使用中位数排除异常切片干扰
            robust_width = np.median(widths)
            stats.append({
                "filename": os.path.basename(f),
                "robust_width": robust_width,
                "matrix_z": img.GetSize()[2],
                "max_area": max_area
            })
            
    return pd.DataFrame(stats)

# 分别分析两个组
# 假设你已经运行了分析函数
df_168 = analyze_dataset_v3(r"C:\Users\xxxxxxyp\Desktop\dlbcl\images")
df_144 = analyze_dataset_v3(r"C:\Users\xxxxxxyp\Desktop\Light-3D-Unet-Front\data\raw\images")

print("--- 168 组统计摘要 ---")
print(df_168[['robust_width', 'matrix_z']].describe().loc[['mean', 'std', 'min', 'max']])

print("\n--- 144 组统计摘要 ---")
print(df_144[['robust_width', 'matrix_z']].describe().loc[['mean', 'std', 'min', 'max']])

# 关键对比：人体像素占比
print(f"\n168组平均人体像素宽度: {df_168['robust_width'].mean():.2f}")
print(f"144组平均人体像素宽度: {df_144['robust_width'].mean():.2f}")