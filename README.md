# AI 看图说话：基于CNN与Transformer的图像字幕生成

本项目是一个基于 PyTorch 实现的深度学习模型，能够为输入的图片自动生成描述性文字（即“图像字幕”）。模型采用了经典的 **编码器-解码器（Encoder-Decoder）** 架构，其中 CNN 作为编码器负责理解图像内容，Transformer 作为解码器负责生成文本序列。

## 📁 项目结构

```
image_captioning/
├── data/                      # 存放原始数据集
├── models/                    # 存放训练好的模型权重 (.pth) 和词汇表 (.pkl)
├── deployment/                # 存放用于 Gradio 部署的应用文件
├── training_logs/             # 存放每次训练的日志 (.txt)
├── prediction_plots/          # 存放 predict.py 生成的可视化结果图
│
├── get_data.py                # [数据获取] 从 Kaggle 自动下载并准备数据集
├── vocabulary.py              # [词汇表] 定义单词-索引映射的 Vocabulary 类
├── data_setup.py              # [数据处理] 构建 Dataset 和 DataLoader
├── model_builder.py           # [模型构建] 定义 CNN 编码器和 Transformer 解码器
├── engine.py                  # [训练引擎] 包含训练/验证循环逻辑
├── utils.py                   # [工具函数] 包含保存/加载模型、词汇表等函数
├── train.py                   # [训练入口] 启动并编排训练流程
├── predict.py                 # [预测入口] 使用训练好的模型生成字幕及可视化
└── README.md                  # 项目说明文件（本文件）
```

## 🛠️ 环境搭建

建议使用 Conda 管理环境：

1. **创建并激活环境**

```bash
conda create -n captioning_env python=3.10 -y
conda activate captioning_env
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **下载 SpaCy 语言模型**

```bash
python -m spacy download en_core_web_sm
```

至此，开发和训练环境已准备就绪。

## 🚀 快速开始

### 1. 获取数据集

Flickr8k 数据集

```bash
python get_data.py
```

该脚本会检查 `data/` 目录，如不存在则自动下载、解压并整理文件。

### 2. 模型训练

启动训练脚本，查看训练/验证损失：

```bash
python train.py
```

- 最优模型自动保存为 `models/captioning_model_best.pth`。
- 训练日志保存在 `training_logs/`。

### 3. 模型预测与评估

训练完成后，可使用 `predict.py` 生成字幕并可视化：

```bash
# 单张图片预测
python predict.py --image_path "path/to/your/image.jpg"

# 随机抽样 n 张图片，并对比真实描述
python predict.py -n 5

# 输出并保存可视化图表
python predict.py -n 5 --plot

# 指定特定模型检查点
python predict.py -n 5 --model_path "models/captioning_model_epoch_10.pth"
```

可视化结果保存在 `prediction_plots/`。

### 4. 部署为 Web 应用

进入部署目录，准备模型文件，然后启动 Gradio 应用：

```bash
cd deployment
cp ../models/captioning_model_best.pth .
cp ../models/vocabulary.pkl .
python app.py
```

程序会输出本地访问地址（如 `http://127.0.0.1:7860`）。

如在 Colab 或远程环境运行，可在 `app.py` 中将 `demo.launch()` 修改为 `demo.launch(share=True)` 以生成公开链接。



