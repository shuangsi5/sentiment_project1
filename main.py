"""
文本情感分类系统 - 主程序
数据集：IMDB 电影评论
使用：BiLSTM with Attention
硬件：NVIDIA RTX 4060 (CUDA 12.9)
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# 导入自定义模块
from utils import load_imdb_from_local, clean_text, build_vocab, encode_dataset, DATA_DIR
from model import BiLSTMSentiment, SimpleBiLSTM

warnings.filterwarnings('ignore')

# 设置随机种子保证可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== 第1步：从本地文件读取数据 ====================
print("=" * 70)
print("文本情感分类实验 - IMDB电影评论")
print("=" * 70)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 设置数据路径
imdb_path = os.path.join(DATA_DIR, "aclImdb")
processed_path = os.path.join(DATA_DIR, "processed")

# 检查是否已有处理好的数据
processed_file = os.path.join(processed_path, "imdb_processed.npz")

if os.path.exists(processed_file):
    print("\n[1/7] 加载已处理的数据...")
    data = np.load(processed_file, allow_pickle=True)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    with open(os.path.join(processed_path, "vocab.pkl"), 'rb') as f:
        import pickle
        word2idx = pickle.load(f)
    vocab_size = len(word2idx)

    print(f"数据加载完成!")
    print(f"词汇表大小: {vocab_size:,}")
    print(f"训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    print(f"测试集: {X_test.shape}")
else:
    print("\n[1/7] 从本地文件读取IMDB数据集...")
    if not os.path.exists(imdb_path):
        print(f"错误: 找不到数据集目录 {imdb_path}")
        print("\n请按以下步骤操作:")
        print("1. 下载 aclImdb_v1.tar.gz")
        print("2. 解压到 ./data/ 目录")
        print("3. 确保目录结构为: ./data/aclImdb/train/pos/")
        sys.exit(1)

    # 加载原始数据
    train_texts, train_labels, test_texts, test_labels = load_imdb_from_local(imdb_path)

    # 保存原始数据以便调试
    os.makedirs(processed_path, exist_ok=True)

    print("\n[2/7] 数据预处理...")
    # 清洗文本
    print("清洗训练集...")
    train_cleaned = [clean_text(t) for t in train_texts]
    print("清洗测试集...")
    test_cleaned = [clean_text(t) for t in test_texts]

    # 构建词汇表
    print("\n构建词汇表...")
    word2idx, vocab_size = build_vocab(train_cleaned, max_size=20000,
                                        save_path=os.path.join(processed_path, "vocab.pkl"))
    print(f"词汇表大小: {vocab_size:,}")

    # 编码数据
    print("\n编码训练集...")
    X_train_all = encode_dataset(train_cleaned, word2idx, verbose=True)
    y_train_all = np.array(train_labels, dtype=np.int64)

    print("\n编码测试集...")
    X_test = encode_dataset(test_cleaned, word2idx, verbose=False)
    y_test = np.array(test_labels, dtype=np.int64)

    # 划分训练集和验证集
    print("\n划分训练集和验证集...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.3, random_state=42, stratify=y_train_all
    )

    # 保存处理好的数据
    print("\n保存处理好的数据...")
    np.savez(processed_file,
             X_train=X_train, X_val=X_val, X_test=X_test,
             y_train=y_train, y_val=y_val, y_test=y_test)

    print(f"\n数据预处理完成!")
    print(f"训练集: {X_train.shape} (正面: {sum(y_train)}, 负面: {len(y_train)-sum(y_train)})")
    print(f"验证集: {X_val.shape} (正面: {sum(y_val)}, 负面: {len(y_val)-sum(y_val)})")
    print(f"测试集: {X_test.shape} (正面: {sum(y_test)}, 负面: {len(y_test)-sum(y_test)})")

# ==================== 第3步：模型初始化 ====================
print("\n" + "=" * 70)
print("[3/7] 模型初始化")
print("=" * 70)

# 模型参数
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
USE_ATTENTION = True

# 选择模型
model = BiLSTMSentiment(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    use_attention=USE_ATTENTION
).to(device)

# 打印模型摘要
model.summary()

# ==================== 第4步：训练配置 ====================
print("\n" + "=" * 70)
print("[4/7] 训练配置")
print("=" * 70)

# 训练参数
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3  # 早停耐心值

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 创建DataLoader
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"优化器: Adam")
print(f"损失函数: CrossEntropyLoss")

# ==================== 第5步：训练模型 ====================
print("\n" + "=" * 70)
print("[5/7] 开始训练模型")
print("=" * 70)

train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = 0.0
best_model_path = "best_model.pth"
patience_counter = 0
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()

    # ---- 训练阶段 ----
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 每50个batch显示进度
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---- 验证阶段 ----
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

    # 学习率调度
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    epoch_time = time.time() - epoch_start_time

    # 打印epoch结果
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] - {epoch_time:.1f}s")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
    print(f"  Learning Rate: {current_lr:.6f}")

    # 早停和保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, best_model_path)
        print(f"  ✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  - 未提升，耐心计数: {patience_counter}/{PATIENCE}")

    # 早停检查
    if patience_counter >= PATIENCE:
        print(f"\n早停触发! 最佳验证准确率: {best_val_acc:.2f}%")
        break

# 加载最佳模型
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
best_epoch = checkpoint['epoch']

total_time = time.time() - start_time
print(f"\n训练完成! 总时间: {total_time:.1f}s")
print(f"最佳模型来自 Epoch {best_epoch}, Val Acc: {best_val_acc:.2f}%")

# ==================== 第6步：测试评估 ====================
print("\n" + "=" * 70)
print("[6/7] 测试集评估")
print("=" * 70)

model.eval()
all_predictions = []
all_true_labels = []
all_probabilities = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# 计算各项指标
accuracy = accuracy_score(all_true_labels, all_predictions)
precision = precision_score(all_true_labels, all_predictions, average='binary')
recall = recall_score(all_true_labels, all_predictions, average='binary')
f1 = f1_score(all_true_labels, all_predictions, average='binary')

# 分类报告
print("\n" + "=" * 60)
print("            📊 详细评估报告")
print("=" * 60)
print(f"\n总体指标:")
print(f"  Accuracy  (准确率):  {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision (精确率):  {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall    (召回率):  {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1 Score  (F1值):    {f1:.4f}  ({f1 * 100:.2f}%)")

# 分类报告
print("\n分类报告:")
print(classification_report(all_true_labels, all_predictions,
                          target_names=['负面', '正面'], digits=4))

# 混淆矩阵
cm = confusion_matrix(all_true_labels, all_predictions)
print("\n混淆矩阵:")
print(f"              预测负面   预测正面")
print(f"实际负面      {cm[0,0]:6d}      {cm[0,1]:6d}")
print(f"实际正面      {cm[1,0]:6d}      {cm[1,1]:6d}")
print("=" * 60)

# ==================== 第7步：可视化结果 ====================
print("\n" + "=" * 70)
print("[7/7] 生成可视化图表")
print("=" * 70)

# 创建可视化目录
vis_dir = "visualizations"
os.makedirs(vis_dir, exist_ok=True)

# 1. 训练曲线
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=5)
plt.plot(val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=5)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axvline(x=best_epoch-1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch {best_epoch}')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, 'b-o', label='Train Acc', linewidth=2, markersize=5)
plt.plot(val_accs, 'r-s', label='Val Acc', linewidth=2, markersize=5)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Training and Validation Accuracy', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axvline(x=best_epoch-1, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch {best_epoch}')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
print(f"✓ 训练曲线保存: {vis_dir}/training_curves.png")

# 2. 混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['负面', '正面'],
            yticklabels=['负面', '正面'])
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.title('混淆矩阵', fontsize=14)
plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
print(f"✓ 混淆矩阵保存: {vis_dir}/confusion_matrix.png")

# 3. 模型性能总结
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

bars = ax.bar(metrics, values, color=colors, alpha=0.8)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('模型性能指标', fontsize=14)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# 在柱子上显示数值
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'performance_metrics.png'), dpi=150, bbox_inches='tight')
print(f"✓ 性能指标图保存: {vis_dir}/performance_metrics.png")

# 显示图片
plt.show()

# ==================== 总结输出 ====================
print("\n" + "=" * 70)
print("🎉 实验完成！")
print("=" * 70)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总运行时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
print(f"\n📁 生成的文件:")
print(f"  ✓ 最佳模型: {best_model_path}")
print(f"  ✓ 可视化图表: {vis_dir}/ 目录")
print(f"  ✓ 处理后的数据: {processed_path}/ 目录")
print(f"\n📊 最终测试结果:")
print(f"  - 准确率: {accuracy*100:.2f}%")
print(f"  - 精确率: {precision*100:.2f}%")
print(f"  - 召回率: {recall*100:.2f}%")
print(f"  - F1值: {f1 * 100:.2f}%")
print("\n💡 模型信息:")
print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"  - 硬件: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 70)
print("\n✅ 实验报告所需的所有数据和图表已生成完成！")