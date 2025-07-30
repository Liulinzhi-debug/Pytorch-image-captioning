# get_data.py
"""
一个用于自动下载和准备 Flickr8k 数据集的辅助脚本。
这个更新版本在移动文件时增加了 tqdm 进度条。

在使用前，请确保您已经安装了 kagglehub 并配置好了您的 Kaggle API 密钥。
1. pip install kagglehub tqdm
2. 将您的 kaggle.json 文件放置在 ~/.kaggle/ 目录下。
"""
import os
import shutil
import pathlib
import kagglehub
import pandas as pd
from tqdm.auto import tqdm

def copy_with_progress(src, dst):
    """
    自定义的复制函数，用于在复制文件时显示tqdm进度条。
    src: 源文件/文件夹路径
    dst: 目标文件/文件夹路径
    """
    # 如果是文件夹，则递归复制
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        files = os.listdir(src)
        pbar = tqdm(files, desc=f"复制到 {dst}", leave=False)
        for file in pbar:
            src_path = os.path.join(src, file)
            dst_path = os.path.join(dst, file)
            shutil.copy2(src_path, dst_path) # copy2 保留元数据
    # 如果是单个文件
    else:
        shutil.copy2(src, dst)

def download_and_setup_flickr8k():
    """
    检查Flickr8k数据集是否存在，如果不存在，则使用kagglehub下载并整理到正确的目录结构。
    """
    project_data_path = pathlib.Path("data/flickr8k")
    images_path = project_data_path / "Images"
    captions_path = project_data_path / "captions.txt"

    if images_path.is_dir() and captions_path.is_file():
        print(f"[INFO] Flickr8k 数据集已存在于 '{project_data_path}'，跳过下载步骤。")
        return

    print("[INFO] Flickr8k 数据集未找到，开始从 Kaggle 下载...")

    try:
        kaggle_extracted_path_str = kagglehub.dataset_download("adityajn105/flickr8k")
        kaggle_extracted_path = pathlib.Path(kaggle_extracted_path_str)
        print(f"[INFO] 数据集已成功下载并解压到缓存: {kaggle_extracted_path}")
    except Exception as e:
        print(f"[ERROR] KaggleHub 下载失败: {e}")
        print("[INFO] 请确认您已正确安装 kagglehub (pip install kagglehub) 并设置了 Kaggle API 密钥。")
        return

    source_images_path = kaggle_extracted_path / "Images"
    source_captions_path = kaggle_extracted_path / "captions.txt"

    if not source_images_path.is_dir() or not source_captions_path.is_file():
        print(f"[ERROR] 在下载的数据中未找到预期的 'Images' 文件夹或 'captions.txt' 文件。")
        return

    project_data_path.mkdir(parents=True, exist_ok=True)

    # --- 使用带进度条的复制代替 shutil.move ---
    print(f"[INFO] 正在将图片复制到 '{images_path}'...")
    copy_with_progress(str(source_images_path), str(images_path))
    
    print(f"[INFO] 正在将描述文件复制到 '{captions_path}'...")
    shutil.copy2(str(source_captions_path), str(captions_path))

    # 清理源文件
    #print("[INFO] 正在清理缓存中的源文件...")
    #shutil.rmtree(source_images_path)
    #os.remove(source_captions_path)
    
    print("[SUCCESS] 数据集已成功准备就绪！")

if __name__ == '__main__':
    download_and_setup_flickr8k()
