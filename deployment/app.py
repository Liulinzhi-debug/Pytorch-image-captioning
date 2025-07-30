# app.py
"""
一个用于图像字幕生成的Gradio Web应用。
"""
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import argparse

# 导入我们项目中的辅助模块
import model_builder
import utils
import vocabulary


# --- 1. 设置 ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义与训练时相同的图像变换
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. 加载词汇表和模型 ---
VOCAB_PATH = "." # 当前目录
MODEL_PATH = "captioning_model_best.pth" # 您可以根据需要更改这个文件名

print("[INFO] 正在加载词汇表和模型...")
vocab = utils.load_vocab(VOCAB_PATH)
vocab_size = len(vocab)

# 实例化一个与训练时配置相同的模型
model = model_builder.EncoderDecoder(
    embed_size=512,
    hidden_size=512,
    vocab_size=vocab_size,
    num_layers=2,
    num_heads=8
).to(device)

# 加载模型权重
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[SUCCESS] 模型加载成功！")
except FileNotFoundError:
    print(f"[ERROR] 模型文件未找到: {MODEL_PATH}")
    gr.Error(f"模型文件 '{MODEL_PATH}' 未找到！请检查部署文件夹。")
    exit()
except Exception as e:
    print(f"[ERROR] 加载模型时发生错误: {e}")
    gr.Error(f"加载模型时出错: {e}")
    exit()

# --- 3. 创建核心预测函数 ---
def predict(image: Image.Image) -> str:
    """
    接收一个PIL图像，返回生成的字幕字符串。
    """
    if image is None:
        return "错误：未提供图片。"

    image_tensor = transform(image).to(device)

    model.eval()
    result_caption_indices = [vocab.stoi["<SOS>"]]
    
    with torch.no_grad():
        features = model.encoder(image_tensor.unsqueeze(0))
        for _ in range(50):
            captions_tensor = torch.LongTensor(result_caption_indices).unsqueeze(0).to(device)
            outputs = model.decoder(features, captions_tensor)
            predicted_index = outputs.argmax(2)[:, -1].item()
            result_caption_indices.append(predicted_index)
            if vocab.itos[predicted_index] == "<EOS>":
                break
    
    result_caption = [vocab.itos[idx] for idx in result_caption_indices]
    return " ".join(result_caption[1:-1])

# --- 4. 创建并启动 Gradio 应用 ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传一张图片"),
    outputs=gr.Textbox(label="AI 生成的图片描述"),
    title="🖼️笨笨 AI 看图说话 📸",
    description="上传任意一张图片，让AI来告诉你它看到了什么。这个模型由CNN编码器和Transformer解码器构成。",
    examples=None,
    allow_flagging="never"
)

if __name__ == "__main__":
    # 添加 share=True 以在Colab中生成公开链接
    demo.launch(share=True)