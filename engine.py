# engine.py
import torch
from tqdm.auto import tqdm

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler):
    model.train()
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    total_loss = 0.0

    for imgs, captions in pbar:
        imgs = imgs.to(device)
        captions = captions.to(device)
        
        captions_input = captions[:, :-1]
        captions_target = captions[:, 1:]

        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            outputs = model(imgs, captions_input)
            loss = loss_fn(outputs.reshape(-1, outputs.shape[2]), captions_target.reshape(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # --- 核心修改：在这里添加梯度裁剪 ---
        # 在优化器更新权重之前，对梯度进行裁剪，防止其过大
        # max_norm=1.0 是一个常用的默认值
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # --- 核心修改结束 ---

        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}", 
            lr=f"{optimizer.param_groups[0]['lr']:.6f}"
        )
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss
