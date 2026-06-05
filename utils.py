"""
工具函数模块：数据清洗、词汇表构建、文本编码
"""

import re
import numpy as np
from collections import Counter

# 全局配置
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 300

def clean_text(text):
    """
    清洗文本：小写、去HTML标签、去标点、去停用词
    """
    try:
        from nltk.corpus import stopwords
        STOP_WORDS = set(stopwords.words('english'))
    except:
        STOP_WORDS = set(['the', 'a', 'an', 'is', 'are', 'was', 'were',
                         'be', 'been', 'being', 'have', 'has', 'had',
                         'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'shall', 'can',
                         'i', 'you', 'he', 'she', 'it', 'we', 'they',
                         'my', 'your', 'his', 'her', 'its', 'our', 'their',
                         'this', 'that', 'these', 'those', 'am', 'in',
                         'on', 'at', 'by', 'for', 'with', 'about',
                         'against', 'between', 'into', 'through', 'during',
                         'before', 'after', 'above', 'below', 'to', 'from',
                         'up', 'down', 'of', 'and', 'but', 'or', 'nor',
                         'not', 'so', 'yet', 'both', 'each', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'only',
                         'own', 'same', 'than', 'too', 'very'])

    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)      # 替换 <br> 为空格
    text = re.sub(r'<.*?>', '', text)           # 去其他HTML标签
    text = re.sub(r'[^a-z\s]', '', text)        # 只保留字母和空格
    words = text.split()
    words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    return ' '.join(words)


def build_vocab(texts, max_size=MAX_VOCAB_SIZE):
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


def encode_dataset(texts, word2idx, max_len=MAX_SEQ_LEN):
    """
    对整个数据集进行编码
    """
    return np.array([encode_text(t, word2idx, max_len) for t in texts])