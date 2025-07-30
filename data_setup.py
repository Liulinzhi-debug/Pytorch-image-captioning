import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from vocabulary import Vocabulary # 导入我们自己的词汇表类

class FlickrDataset(Dataset):
    """
    FlickrDataset 类

    Args:
        root_dir (str): 图像文件夹的路径。
        df (pd.DataFrame): 包含图像文件名和对应描述的DataFrame。
        vocab (Vocabulary): 预先构建好的词汇表对象。
        transform (callable, optional): 应用于图像的变换。
    """
    def __init__(self, root_dir, df, vocab, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.transform = transform
        self.vocab = vocab

        # 直接从传入的dataframe获取数据
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 使用 .iloc 来安全地索引可能经过筛选的DataFrame
        caption = self.captions.iloc[index]
        img_id = self.imgs.iloc[index]
        img_path = os.path.join(self.root_dir, img_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets

def create_dataloaders(
    root_folder, 
    annotation_file, 
    train_transform,
    val_transform,
    batch_size=32, 
    val_split=0.15,
    freq_threshold=5,
    num_workers=0, 
    shuffle=True, 
    pin_memory=True
):
    """
    创建训练和验证数据加载器。

    该函数首先在整个数据集上构建词汇表，然后根据唯一的图像ID将数据集
    划分为训练集和验证集，确保同一图像的所有标题都在同一个集合中。

    Args:
        root_folder (str): 图像文件夹的根目录。
        annotation_file (str): 包含图像文件名和标题的 .txt 文件的路径。
        train_transform (callable): 应用于训练图像的变换（通常包含数据增强）。
        val_transform (callable): 应用于验证图像的变换（通常不包含数据增强）。
        batch_size (int): 每个批次中的样本数。
        val_split (float): 用于验证集的唯一图像的比例（例如，0.15 表示 15%）。
        freq_threshold (int): 在构建词汇表时，一个词必须出现的最小次数。
        num_workers (int): 用于数据加载的子进程数。
        shuffle (bool): 是否在每个 epoch 开始时打乱训练数据。
        pin_memory (bool): 如果为 True，数据加载器会将张量复制到 CUDA 固定内存中。

    Returns:
        tuple: 包含 train_loader, val_loader, 和 vocab 的元组。
    """
    # 1. 加载所有标注并构建词汇表
    df = pd.read_csv(annotation_file)
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(df["caption"].tolist())
    pad_idx = vocab.stoi["<PAD>"]

    # 2. 基于唯一的图像ID进行训练/验证集划分
    unique_images = df["image"].unique()
    np.random.shuffle(unique_images)
    split_idx = int(len(unique_images) * (1 - val_split))
    
    train_image_ids = unique_images[:split_idx]
    val_image_ids = unique_images[split_idx:]

    train_df = df[df["image"].isin(train_image_ids)].reset_index(drop=True)
    val_df = df[df["image"].isin(val_image_ids)].reset_index(drop=True)

    # 3. 为训练集和验证集创建 Dataset 实例
    train_dataset = FlickrDataset(root_folder, train_df, vocab, transform=train_transform)
    val_dataset = FlickrDataset(root_folder, val_df, vocab, transform=val_transform)

    # 4. 创建 Collate 函数和 Dataloaders
    collate_fn = MyCollate(pad_idx=pad_idx)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False, # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, vocab
