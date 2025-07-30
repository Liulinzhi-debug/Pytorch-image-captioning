import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import data_setup
import model_builder
import utils

from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

# --- 超参数 ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 40
FREQ_THRESHOLD = 5
EMBED_SIZE = 512
HIDDEN_SIZE = 512
NUM_LAYERS = 2
NUM_HEADS = 8
ROOT_DIR = "data/flickr8k/Images"
CAPTIONS_FILE = "data/flickr8k/captions.txt"

# --- 设置 ---
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# --- 核心修改：为训练集和验证集定义不同的图像变换 ---
# 训练集变换：包含数据增强，提高模型泛化能力
train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomCrop((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集变换：不进行数据增强，确保评估的一致性
val_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# --- 核心修改结束 ---

# --- 数据加载 ---
train_loader, val_loader, vocab = data_setup.create_dataloaders(
    root_folder=ROOT_DIR,
    annotation_file=CAPTIONS_FILE,
    train_transform=train_transform,
    val_transform=val_transform,
    batch_size=BATCH_SIZE,
    freq_threshold=FREQ_THRESHOLD
)
vocab_size = len(vocab)
pad_idx = vocab.stoi["<PAD>"]

# --- 模型、损失、优化器与调度器 ---
model = model_builder.EncoderDecoder(
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    vocab_size=vocab_size,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.amp.GradScaler()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3
)

# --- 新增：准确率计算函数 ---
def accuracy_fn(y_pred, y_true, pad_idx):
    """
    计算词元级别的准确率，忽略填充部分。
    Args:
        y_pred (torch.Tensor): 模型的原始输出 (logits)，形状为 [batch_size * seq_len, vocab_size]。
        y_true (torch.Tensor): 真实标签，形状为 [batch_size * seq_len]。
        pad_idx (int): <PAD> 标记在词汇表中的索引。
    Returns:
        float: 准确率。
    """
    # 获取预测最准确的词的索引
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    
    # 创建一个 mask 来忽略 <PAD> 词元
    non_pad_elements = (y_true != pad_idx)
    
    # 计算在非填充位置上预测正确的数量
    correct = torch.sum((y_pred_class == y_true) & non_pad_elements)
    
    # 计算总的非填充词元数量
    total_non_pad = torch.sum(non_pad_elements)
    
    # 计算准确率
    accuracy = correct.item() / total_non_pad.item()
    return accuracy
# --- 新增结束 ---


# --- 训练准备 ---
results = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}
best_val_loss = float('inf')

# --- 训练与验证循环 ---
for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    
    # --- 训练步骤 ---
    model.train()
    train_loss, train_acc = 0, 0
    pbar_train = tqdm(train_loader, desc="Training", leave=False)
    for imgs, captions in pbar_train:
        imgs = imgs.to(device)
        captions = captions.to(device)
        
        captions_input = captions[:, :-1]
        captions_target = captions[:, 1:]

        with torch.cuda.amp.autocast():
            outputs = model(imgs, captions_input)
            loss = loss_fn(outputs.reshape(-1, outputs.shape[2]), captions_target.reshape(-1))
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        acc = accuracy_fn(
            y_pred=outputs.reshape(-1, outputs.shape[2]),
            y_true=captions_target.reshape(-1),
            pad_idx=pad_idx
        )
        train_loss += loss.item()
        train_acc += acc
        pbar_train.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    
    # --- 验证步骤 ---
    model.eval()
    val_loss, val_acc = 0, 0
    pbar_val = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for imgs, captions in pbar_val:
            imgs = imgs.to(device)
            captions = captions.to(device)
            captions_input = captions[:, :-1]
            captions_target = captions[:, 1:]

            with torch.cuda.amp.autocast():
                outputs = model(imgs, captions_input)
                loss = loss_fn(outputs.reshape(-1, outputs.shape[2]), captions_target.reshape(-1))
            
            acc = accuracy_fn(
                y_pred=outputs.reshape(-1, outputs.shape[2]),
                y_true=captions_target.reshape(-1),
                pad_idx=pad_idx
            )
            val_loss += loss.item()
            val_acc += acc
            pbar_val.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {avg_train_loss:.4f} | train_acc: {avg_train_acc:.4f} | "
        f"val_loss: {avg_val_loss:.4f} | val_acc: {avg_val_acc:.4f}"
    )

    # --- 更新调度器并保存最佳模型 ---
    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        print(f"Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving best model...")
        best_val_loss = avg_val_loss
        utils.save_model(model, "models", "captioning_model_best.pth")

    # 记录结果
    results["train_loss"].append(avg_train_loss)
    results["train_acc"].append(avg_train_acc)
    results["val_loss"].append(avg_val_loss)
    results["val_acc"].append(avg_val_acc)

# --- 训练结束后 ---
print("\n[INFO] 所有训练世代已完成。")
utils.save_vocab(vocab, "models") 
utils.save_model(model, "models", "captioning_model_final.pth")
print("[SUCCESS] 最终模型、最佳模型和词汇表已保存。")

# --- 将训练摘要保存到 .txt 文件 ---
log_dir = "training_logs"
Path(log_dir).mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"training_summary_{timestamp}.txt"
log_filepath = Path(log_dir) / log_filename

try:
    with open(log_filepath, 'w') as f:
        print(f"\n[INFO] 正在将训练摘要保存到: {log_filepath}...")
        
        f.write("--- Hyperparameters ---\n")
        f.write(f"MODEL: EncoderDecoder (CNN + Transformer)\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"INITIAL_LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"EMBED_SIZE: {EMBED_SIZE}\n")
        f.write(f"HIDDEN_SIZE: {HIDDEN_SIZE}\n")
        f.write(f"DEVICE: {device}\n\n")
        
        f.write("--- Epoch Results ---\n")
        for i in range(len(results["train_loss"])):
            summary_line = (
                f"Epoch: {i+1} | "
                f"train_loss: {results['train_loss'][i]:.4f} | "
                f"train_acc: {results['train_acc'][i]:.4f} | "
                f"val_loss: {results['val_loss'][i]:.4f} | "
                f"val_acc: {results['val_acc'][i]:.4f}\n"
            )
            f.write(summary_line)
            
        print("[SUCCESS] 训练摘要已成功保存。")

except Exception as e:
    print(f"[ERROR] 保存训练摘要时发生错误: {e}")
