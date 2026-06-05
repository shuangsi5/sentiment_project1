"""
BiLSTM 模型定义模块
"""

import torch
import torch.nn as nn


class BiLSTMSentiment(nn.Module):
    """
    双向LSTM情感分类模型
    架构: Embedding → BiLSTM → Dropout → FC(64) → Output(2)
    """

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 num_layers=2, num_classes=2, dropout=0.5):
        super(BiLSTMSentiment, self).__init__()

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
        self.fc = nn.Linear(hidden_dim * 2, 64)  # 双向所以 hidden_dim * 2
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # LSTM 前向传播
        lstm_out, (hidden, cell) = self.embedded(embedded)
        # hidden: [num_layers * 2, batch, hidden_dim]

        # 取最后一层的双向隐藏状态
        hidden_fwd = hidden[-2, :, :]  # [batch, hidden_dim]
        hidden_bwd = hidden[-1, :, :]  # [batch, hidden_dim]
        combined = torch.cat((hidden_fwd, hidden_bwd), dim=1)  # [batch, hidden_dim*2]

        # 全连接层
        out = torch.relu(self.fc(combined))
        out = self.dropout(out)
        out = self.out(out)

        return out