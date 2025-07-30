# predict.py
import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import random
import pandas as pd # 需要安装 pandas: pip install pandas
import numpy as np # 新增：导入numpy

# --- 新增：导入可视化和文本处理库 ---
import matplotlib.pyplot as plt
import textwrap
# --- 新增结束 ---

import model_builder
import utils

def caption_single_image(image_path, model, vocab, transform, device):
    """加载、预处理并为单张图片生成字幕。"""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"错误：无法打开图片 {image_path}。 {e}")
        return None

    # 应用变换
    image_tensor = transform(image).to(device)

    # 使用模型生成字幕索引
    model.eval()
    result_caption_indices = [vocab.stoi["<SOS>"]]
    
    with torch.no_grad():
        features = model.encoder(image_tensor.unsqueeze(0))
        for _ in range(50): # 最大长度为50
            captions_tensor = torch.LongTensor(result_caption_indices).unsqueeze(0).to(device)
            
            outputs = model.decoder(features, captions_tensor)
            predicted_index = outputs.argmax(2)[:, -1].item()
            
            result_caption_indices.append(predicted_index)

            if vocab.itos[predicted_index] == "<EOS>":
                break
    
    # 将索引转换回单词
    result_caption = [vocab.itos[idx] for idx in result_caption_indices]
    
    # 过滤掉特殊字符并拼接成句子
    return " ".join(result_caption[1:-1]) # 去掉 <SOS> 和 <EOS>

# --- 新增：可视化函数 ---
def display_prediction_with_image(image_path, generated_caption, ground_truths, output_dir="prediction_plots"):
    """
    创建一个包含图片、生成字幕和真实字幕的可视化图表，并保存到文件。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始图片
    image = Image.open(image_path).convert("RGB")
    
    # 设置绘图
    plt.figure(figsize=(12, 9))
    plt.imshow(image)
    plt.axis('off') # 不显示坐标轴

    # 将生成的字幕作为标题，并自动换行
    wrapped_title = "\n".join(textwrap.wrap(f"AI 生成的字幕: {generated_caption}", width=60))
    plt.title(wrapped_title, fontsize=14, pad=20)

    # 将真实字幕放在图片下方
    gt_text = "真实的字幕:\n" + "\n".join([f"- {gt}" for gt in ground_truths])
    
    # 使用 figtext 在图表下方添加文本
    plt.figtext(0.5, 0.01, gt_text, ha="center", fontsize=10, va="bottom", wrap=True)
    
    # 调整布局以防止文本重叠
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # 留出底部空间给真实字幕

    # 保存图像
    img_filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"prediction_{img_filename}.png")
    plt.savefig(save_path)
    plt.close() # 关闭图表，释放内存
    print(f"   -> 可视化结果已保存至: {save_path}")
# --- 新增结束 ---

if __name__ == "__main__":
    # --- 1. 修改命令行参数解析 ---
    parser = argparse.ArgumentParser(description="为一个或多个图片生成字幕，并可选择进行可视化。")
    
    parser.add_argument("--image_path", type=str, help="要生成字幕的单张图片路径。")
    parser.add_argument("-n", "--num_random", type=int, help="从数据集中随机抽取N张图片进行预测和对比。")
    
    # --- 新增：可视化开关 ---
    parser.add_argument("--plot", action="store_true", help="生成并保存包含图片和字幕的可视化结果。")
    
    # --- 新增：随机种子参数 ---
    parser.add_argument("--seed", type=int, default=42, help="设置随机种子以保证结果可复现。") #
    # --- 新增结束 ---
    
    parser.add_argument("--model_path", type=str, default="deployment/captioning_model_final.pth", help="已训练好的模型权重路径。")
    parser.add_argument("--vocab_path", type=str, default="models/", help="已保存的词汇表所在目录。")
    parser.add_argument("--image_dir", type=str, default="/home/amax/student_su/image_captioning/data/flickr8k/Images", help="随机抽样时使用的图片目录。")
    parser.add_argument("--captions_file", type=str, default="/home/amax/student_su/image_captioning/data/flickr8k/captions.txt", help="随机抽样时用于对比的真实字幕文件。")
    
    args = parser.parse_args()

    # --- 2. 检查模式并执行 ---
    if not args.image_path and not args.num_random:
        parser.error("错误：请至少提供一个操作模式。使用 --image_path 指定图片，或使用 -n/--num_random 进行随机抽样。")

    # --- 3. 公共设置与加载 ---
    
    # --- 新增：设置随机种子 ---
    random.seed(args.seed) #
    np.random.seed(args.seed) #
    torch.manual_seed(args.seed) #
    torch.cuda.manual_seed_all(args.seed) #
    # --- 新增结束 ---
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"[INFO] 正在加载词汇表和模型... (随机种子: {args.seed})") #
    vocab = utils.load_vocab(args.vocab_path)
    vocab_size = len(vocab)
    
    model = model_builder.EncoderDecoder(
        embed_size=512,
        hidden_size=512,
        vocab_size=vocab_size,
        num_layers=2,
        num_heads=8
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(f"[SUCCESS] 模型 '{args.model_path}' 加载成功。")
    except FileNotFoundError:
        print(f"[ERROR] 模型文件未找到: {args.model_path}")
        exit()
        
    # --- 新增：如果需要对比或画图，提前加载字幕文件 ---
    captions_df = None
    if args.plot or args.num_random:
        try:
            captions_df = pd.read_csv(args.captions_file)
        except FileNotFoundError:
            print(f"[ERROR] 字幕文件未找到，无法进行对比或绘图: {args.captions_file}")
            exit()
    # --- 新增结束 ---


    # --- 4. 根据模式执行预测 ---
    if args.image_path:
        generated_caption = caption_single_image(args.image_path, model, vocab, transform, device)
        if generated_caption:
            print("\n--- 模型预测结果 ---")
            print(f"图片: {args.image_path}")
            print(f"  -> 生成的字幕: {generated_caption}")
            
            # 如果需要绘图，查找真实字幕并调用绘图函数
            if args.plot and captions_df is not None:
                img_name = os.path.basename(args.image_path)
                ground_truths = captions_df[captions_df["image"] == img_name]["caption"].tolist()
                display_prediction_with_image(args.image_path, generated_caption, ground_truths)

    if args.num_random:
        print(f"\n[INFO] 进入随机抽样模式，将从 '{args.image_dir}' 中抽取 {args.num_random} 张图片...")
        all_images = os.listdir(args.image_dir)
        random_images = random.sample(all_images, args.num_random) #

        for i, img_name in enumerate(random_images):
            img_path = os.path.join(args.image_dir, img_name)
            generated_caption = caption_single_image(img_path, model, vocab, transform, device)
            ground_truths = captions_df[captions_df["image"] == img_name]["caption"].tolist()
            
            print(f"\n--- 预测样本 {i+1}/{args.num_random} ---")
            print(f"图片: {img_path}")
            print(f"  -> 生成的字幕: {generated_caption}")
            
            if ground_truths:
                print("  -> 真实的字幕:")
                for gt in ground_truths:
                    print(f"        - {gt}")
            
            # 如果需要绘图，调用绘图函数
            if args.plot:
                display_prediction_with_image(img_path, generated_caption, ground_truths)
# 指定模式
# python predict.py --image_path "data/flickr8k/Images/some_specific_image.jpg"

# 随机抽样对比模式
# python predict.py -n 5

# 您也可以在使用随机模式时，指定不同的模型文件进行对比，例如
# python predict.py -n 5 --model_path "models/captioning_model_final.pth"

# --plot 可视化