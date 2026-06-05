"""
文本情感分类系统 - 主程序（使用本地数据集）
数据集：IMDB 电影评论（从本地文件读取）
使用：BiLSTM 模型
"""

import os
import re
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== 第1步：从本地文件读取数据 ====================
print("=" * 60)
print("第1步：从本地文件读取 IMDB 数据集")
print("=" * 60)

def read_imdb_from_folder(data_path):
    """从本地文件夹读取 IMDB 数据集"""
    texts = []
    labels = []

    for split in ['train', 'test']:
        for label, label_name in [(1, 'pos'), (0, 'neg')]:
            folder_path = os.path.join(data_path, split, label_name)
            if not os.path.exists(folder_path):
                print(f"警告：找不到文件夹 {folder_path}")
                continue

            file_count = 0
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        texts.append(text)
                        labels.append(label)
                        file_count += 1
                    except Exception as e:
                        print(f"读取文件 {file_path} 出错: {e}")

            print(f"  读取 {split}/{label_name}: {file_count} 个文件")

    return texts, labels

# 设置数据路径
data_path = os.path.join(os.path.dirname(__file__), 'data', 'aclImdb')
if not os.path.exists(data_path):
    # 尝试其他可能的路径
    alt_path = os.path.join(os.path.dirname(__file__), 'data')
    if os.path.exists(alt_path):
        # 检查是否有子目录
        subdirs = [d for d in os.listdir(alt_path) if os.path.isdir(os.path.join(alt_path, d))]
        if subdirs:
            data_path = os.path.join(alt_path, subdirs[0])
            print(f"使用数据路径: {data_path}")
        else:
            print(f"错误：在 {alt_path} 下没有找到数据子目录")
            print("\n请按以下步骤操作：")
            print("1. 下载 aclImdb_v1.tar.gz")
            print("2. 解压到 E:\\PycharmProjects\\sentiment_project\\data\\ 目录")
            print("3. 确保目录结构为：data/aclImdb/train/pos/ 和 data/aclImdb/train/neg/")
            exit(1)
    else:
        print(f"错误：找不到数据目录 {data_path}")
        print("\n请按以下步骤操作：")
        print("1. 下载 aclImdb_v1.tar.gz")
        print("2. 解压到 E:\\PycharmProjects\\sentiment_project\\data\\ 目录")
        print("3. 确保目录结构为：data/aclImdb/train/pos/ 和 data/aclImdb/train/neg/")
        exit(1)

print(f"从 {data_path} 读取数据...")
all_texts, all_labels = read_imdb_from_folder(data_path)

# 分割训练集和测试集（IMDB 已经分好了）
train_texts = []
train_labels = []
test_texts = []
test_labels = []

for i, (text, label) in enumerate(zip(all_texts, all_labels)):
    if i < 25000:  # 前25000条是训练集
        train_texts.append(text)
        train_labels.append(label)
    else:  # 后25000条是测试集
        test_texts.append(text)
        test_labels.append(label)

print(f"\n训练集大小: {len(train_texts)}")
print(f"测试集大小: {len(test_texts)}")
print(f"正样本数（训练集）: {sum(train_labels)}")
print(f"负样本数（训练集）: {len(train_labels) - sum(train_labels)}")

# ==================== 第2步：数据预处理 ====================
print("\n" + "=" * 60)
print("第2步：数据预处理")
print("=" * 60)

# 简单的停用词列表
STOP_WORDS = set([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'i', 'you', 'he', 'she',
    'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
    'this', 'that', 'these', 'those', 'am', 'in', 'on', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'of',
    'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
    'than', 'too', 'very'
])

def clean_text(text):
    """清洗文本"""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    return ' '.join(words)

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 300

print("清洗训练集...")
train_cleaned = [clean_text(t) for t in train_texts]
print("清洗测试集...")
test_cleaned = [clean_text(t) for t in test_texts]

# 构建词汇表
print("构建词汇表...")
all_words = []
for text in train_cleaned:
    all_words.extend(text.split())

word_counts = Counter(all_words)
most_common = word_counts.most_common(MAX_VOCAB_SIZE - 2)
word2idx = {'<PAD>': 0, '<UNK>': 1}
for idx, (word, _) in enumerate(most_common):
    word2idx[word] = idx + 2
vocab_size = len(word2idx)
print(f"词汇表大小: {vocab_size}")

def encode_text(text, max_len=MAX_SEQ_LEN):
    """将文本编码为索引序列"""
    tokens = text.split()[:max_len]
    encoded = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
    encoded = encoded + [0] * (max_len - len(encoded))
    return np.array(encoded, dtype=np.int64)

print("编码数据集...")
X_train_all = np.array([encode_text(t) for t in train_cleaned])
y_train_all = np.array(train_labels)
X_test = np.array([encode_text(t) for t in test_cleaned])
y_test = np.array(test_labels)

# 划分训练集和验证集（7:3）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.3, random_state=42, stratify=y_train_all
)

print(f"\n最终数据划分:")
print(f"  训练集: {X_train.shape[0]} 条")
print(f"  验证集: {X_val.shape[0]} 条")
print(f"  测试集: {X_test.shape[0]} 条")

# ==================== 第3步：定义 BiLSTM 模型 ====================
class BiLSTMSentiment(nn.Module):
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
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        combined = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        out = torch.relu(self.fc(combined))
        out = self.dropout(out)
        out = self.out(out)
        return out

# ==================== 第4步：训练配置 ====================
print("\n" + "=" * 60)
print("第4步：训练配置")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = BiLSTMSentiment(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

BATCH_SIZE = 64
EPOCHS = 8

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==================== 第5步：训练 ====================
print("\n" + "=" * 60)
print("第5步：开始训练")
print("=" * 60)

best_val_acc = 0.0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    print(f'Epoch [{epoch+1}/{EPOCHS}] '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)')

print("\n训练完成！")

# ==================== 第6步：测试评估 ====================
print("\n" + "=" * 60)
print("第6步：测试评估")
print("=" * 60)

if os.path.exists('best_model.pth'):
    model.load_state_dict(torch.load('best_model.pth'))
    print("已加载最佳模型")

model.eval()
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.numpy())

accuracy = accuracy_score(all_true_labels, all_predictions)
precision = precision_score(all_true_labels, all_predictions, average='binary')
recall = recall_score(all_true_labels, all_predictions, average='binary')
f1 = f1_score(all_true_labels, all_predictions, average='binary')

print("\n" + "=" * 48)
print("        📊 测试集评估结果")
print("=" * 48)
print(f"  Accuracy  (准确率):  {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision (精确率):  {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall    (召回率):  {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1 Score  (F1值):    {f1:.4f}  ({f1 * 100:.2f}%)")
print("=" * 52)

# ==================== 第7步：绘制训练曲线 ====================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, 'b-o', label='Train Loss', linewidth=2)
plt.plot(range(1, EPOCHS+1), val_losses, 'r-s', label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_accs, 'b-o', label='Train Acc', linewidth=2)
plt.plot(range(1, EPOCHS+1), val_accs, 'r-s', label='Val Acc', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("\n训练曲线已保存为 training_curves.png")
plt.show()

print("\n" + "=" * 44)
print("🎉 所有步骤已完成！")
print("=" * 46)