# utils.py
"""
包含项目所需的各种辅助函数，
例如保存/加载模型和词汇表。
"""
import torch
from pathlib import Path
import pickle

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """保存一个 PyTorch 模型的状态字典。"""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name
    print(f"[INFO] 正在保存模型至: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def save_vocab(vocab, path, filename="vocabulary.pkl"):
    """使用 pickle 保存 Vocabulary 对象。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vocab_save_path = path / filename
    print(f"[INFO] 正在保存词汇表至: {vocab_save_path}")
    with open(vocab_save_path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path, filename="vocabulary.pkl"):
    """使用 pickle 加载 Vocabulary 对象。"""
    path = Path(path)
    vocab_load_path = path / filename
    print(f"[INFO] 正在从 {vocab_load_path} 加载词汇表...")
    with open(vocab_load_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab