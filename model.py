"""
BiLSTM 模型定义模块（针对 RTX 4060 优化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMSentiment(nn.Module):
    """
    双向LSTM情感分类模型（优化版）
    架构: Embedding → Dropout → BiLSTM → Attention → FC → Output
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.5, use_attention=True):
        super(BiLSTMSentiment, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 注意力机制
        if use_attention:
            self.attention_weights = nn.Linear(hidden_dim * 2, 1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.orthogonal_(self.embedding.weight[1:])  # 跳过padding

    def attention(self, lstm_output):
        """注意力机制"""
        attention_energies = self.attention_weights(lstm_output).squeeze(2)
        attention_weights = F.softmax(attention_energies, dim=1)

        # 应用注意力权重
        context_vector = torch.bmm(lstm_output.transpose(1, 2),
                                   attention_weights.unsqueeze(2)).squeeze(2)
        return context_vector

    def forward(self, x):
        # x: [batch_size, seq_len]

        # 嵌入层
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding_dropout(embedded)

        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        # hidden: [num_layers * 2, batch_size, hidden_dim]

        if self.use_attention:
            # 使用注意力机制
            context = self.attention(lstm_out)
        else:
            # 使用最后时刻的隐藏状态
            hidden_fwd = hidden[-2, :, :]  # 正向最后一层
            hidden_bwd = hidden[-1, :, :]  # 反向最后一层
            context = torch.cat((hidden_fwd, hidden_bwd), dim=1)

        # 分类器
        output = self.classifier(context)

        return output

    def summary(self):
        """打印模型摘要"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 50)
        print("模型结构: BiLSTM with Attention")
        print("=" * 50)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"嵌入维度: {self.embedding_dim}")
        print(f"隐藏层维度: {self.hidden_dim}")
        print(f"LSTM层数: {self.num_layers}")
        print(f"使用注意力: {self.use_attention}")
        print("=" * 50)


class SimpleBiLSTM(nn.Module):
    """
    简化的BiLSTM模型（用于对比实验）
    """
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 num_layers=2, num_classes=2, dropout=0.5):
        super(SimpleBiLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 取最后时刻的输出
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        combined = torch.cat((hidden_fwd, hidden_bwd), dim=1)

        out = F.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)

        return out