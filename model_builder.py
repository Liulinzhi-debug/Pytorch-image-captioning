# model_builder.py
import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    """
    CNN编码器。这个版本开启了微调（fine-tuning），以提取更具针对性的视觉特征。
    """
    def __init__(self, embed_size, train_cnn=True): # 默认开启微调
        super(CNNEncoder, self).__init__()
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # --- 核心修改：只冻结CNN的极早期层，微调大部分后期层 ---
        # 默认情况下，所有参数都是可训练的
        for name, param in self.inception.named_parameters():
            # 我们只冻结模型最开始的几个卷积层
            if "Conv2d_1a_3x3" in name or "Conv2d_2a_3x3" in name or "Conv2d_2b_3x3" in name:
                param.requires_grad = False
            else:
                # 确保模型的其余部分（包括我们新加的fc层）都是可训练的
                param.requires_grad = True

    def forward(self, images):
        # Inception-v3 在训练模式下会返回辅助输出，我们需要设置为评估模式来获取单一输出
        # 或者在训练时显式处理元组输出
        if self.training:
            # InceptionV3's forward method returns a named tuple InceptionOutputs
            # with two main outputs: logits and aux_logits. We only need the main output.
            outputs = self.inception(images)
            if isinstance(outputs, tuple): # For older torchvision versions
                 outputs = outputs[0]
            # For newer torchvision versions, it might be a named tuple
            elif hasattr(outputs, 'logits'):
                 outputs = outputs.logits
            return self.dropout(self.relu(outputs))
        else: # eval mode
            features = self.inception(images)
            return self.dropout(self.relu(features))


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, embed_size)) # 假设最大长度512
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        seq_length = captions.shape[1]
        captions_embedded = self.embedding(captions)
        
        # 添加位置编码
        positions = self.positional_encoding[:, :seq_length, :]
        captions_with_pos = self.dropout(captions_embedded + positions)
        
        # 创建目标序列的注意力掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length).to(captions.device)

        # 解码
        # features 需要 unsqueeze(0) 来匹配 (S, N, E) 的期望形状
        # captions_with_pos 需要 permute(1, 0, 2)
        output = self.transformer_decoder(
            tgt=captions_with_pos.permute(1, 0, 2),
            memory=features.unsqueeze(0),
            tgt_mask=tgt_mask
        )
        
        # 调整回 (N, S, E) 并送入线性层
        output = self.linear(output.permute(1, 0, 2))
        return output

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, num_heads):
        super(EncoderDecoder, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = TransformerDecoder(embed_size, hidden_size, vocab_size, num_layers, num_heads)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    def caption_image(self, image, vocabulary, max_length=50):
        self.eval() # 确保模型处于评估模式
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image.unsqueeze(0)).unsqueeze(0) # unsqueeze(0) for batch
            states = None

            # 初始输入是 <SOS> token
            word_indices = [vocabulary.stoi["<SOS>"]]
            
            for _ in range(max_length):
                captions_tensor = torch.LongTensor(word_indices).unsqueeze(0).to(image.device)
                
                # 预测下一个词
                predictions = self.decoder(x, captions_tensor) # 注意：这里与训练时的调用方式不同
                
                # predictions 的形状是 (1, seq_len, vocab_size)，我们只取最后一个词的预测
                predicted_index = predictions.argmax(2)[:, -1].item()
                
                word_indices.append(predicted_index)
                
                # 如果预测到 <EOS> token，则停止
                if vocabulary.itos[predicted_index] == "<EOS>":
                    break
            
        return [vocabulary.itos[idx] for idx in word_indices]