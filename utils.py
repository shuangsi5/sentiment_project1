"""
工具函数模块：数据清洗、词汇表构建、文本编码
"""

import re
import numpy as np
from collections import Counter
import os
import pickle

# 全局配置
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 300
DATA_DIR = "./data"

def ensure_dir(directory):
    """确保目录存在"""
    os.makedirs(directory, exist_ok=True)

def load_stopwords():
    """加载停用词列表"""
    return set([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'i', 'you', 'he', 'she',
        'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'this', 'that', 'these', 'those', 'am', 'in', 'on', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'of',
        'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
        'than', 'too', 'very', 's', 't', 'don', 'just', 'doesn', 'didn',
        'won', 'wouldn', 'couldn', 'shouldn', 'mightn', 'mustn', 'isn',
        'aren', 'wasn', 'weren', 'haven', 'hasn', 'hadn', 'don'
    ])

def clean_text(text, stop_words=None):
    """
    清洗文本：小写、去HTML标签、去标点、去停用词
    """
    if stop_words is None:
        stop_words = load_stopwords()

    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)      # 替换 <br> 为空格
    text = re.sub(r'<.*?>', '', text)           # 去其他HTML标签
    text = re.sub(r'[^a-z\s]', '', text)        # 只保留字母和空格
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in stop_words]
    return ' '.join(words)

def build_vocab(texts, max_size=MAX_VOCAB_SIZE, save_path=None):
    """
    从文本列表构建词汇表
    返回: word2idx (字典), vocab_size (int)
    """
    all_words = []
    for text in texts:
        all_words.extend(text.split())

    word_counts = Counter(all_words)
    most_common = word_counts.most_common(max_size - 2)  # 留出 PAD 和 UNK

    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(most_common):
        word2idx[word] = idx + 2

    vocab_size = len(word2idx)

    # 保存词汇表
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, 'wb') as f:
            pickle.dump(word2idx, f)
        print(f"词汇表已保存到: {save_path}")

    return word2idx, vocab_size

def load_vocab(vocab_path):
    """加载词汇表"""
    with open(vocab_path, 'rb') as f:
        word2idx = pickle.load(f)
    return word2idx, len(word2idx)

def encode_text(text, word2idx, max_len=MAX_SEQ_LEN):
    """
    将文本转换为索引序列，并进行填充
    """
    tokens = text.split()[:max_len]
    encoded = [word2idx.get(w, word2idx['<UNK>']) for w in tokens]
    # 填充到固定长度
    encoded = encoded + [0] * (max_len - len(encoded))
    return np.array(encoded, dtype=np.int64)

def encode_dataset(texts, word2idx, max_len=MAX_SEQ_LEN, verbose=True):
    """
    对整个数据集进行编码
    """
    encoded = []
    total = len(texts)

    for i, text in enumerate(texts):
        if verbose and i % 1000 == 0:
            print(f"编码进度: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
        encoded.append(encode_text(text, word2idx, max_len))

    return np.array(encoded)

def load_imdb_from_local(data_path):
    """
    从本地文件加载 IMDB 数据集
    """
    def read_files(folder_path, label):
        texts = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                texts.append((text, label))
        return texts

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    # 读取训练集
    print("读取训练集正面评论...")
    train_pos = read_files(os.path.join(data_path, 'train', 'pos'), 1)
    print("读取训练集负面评论...")
    train_neg = read_files(os.path.join(data_path, 'train', 'neg'), 0)

    # 读取测试集
    print("读取测试集正面评论...")
    test_pos = read_files(os.path.join(data_path, 'test', 'pos'), 1)
    print("读取测试集负面评论...")
    test_neg = read_files(os.path.join(data_path, 'test', 'neg'), 0)

    # 合并
    for text, label in train_pos + train_neg:
        train_texts.append(text)
        train_labels.append(label)

    for text, label in test_pos + test_neg:
        test_texts.append(text)
        test_labels.append(label)

    print(f"训练集: {len(train_texts)} 条")
    print(f"测试集: {len(test_texts)} 条")
    print(f"正样本: {sum(train_labels)} 条")
    print(f"负样本: {len(train_labels) - sum(train_labels)} 条")

    return train_texts, train_labels, test_texts, test_labels