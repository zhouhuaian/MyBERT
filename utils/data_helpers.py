import os
import time
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Vocab:
    """
    根据本地的vocab文件，构造词表——微调预训练模型时无需从训练数据创建词表
    """

    UNK = "[UNK]"

    def __init__(self, vocab_path):
        self.stoi = {}  # 字典，记录词和索引的键值对
        self.itos = []  # 列表，记录词表中所有词
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, token in enumerate(f):
                token = token.strip("\n")
                self.stoi[token] = idx
                self.itos.append(token)

    def __getitem__(self, token):
        """
        获取token的索引，支持vocab[token]的方式访问
        """
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        """
        获取词表长度，支持len(vocab)的方式访问
        """
        return len(self.itos)


def pad_sequence(sequences, padding_value=0, max_len=None, batch_first=False):
    """
    对一个序列的样本进行填充
    :param sequences: 一批序列
    :param padding_value: 填充值
    :param max_len: 最大序列长度，以该长度填充序列，若`==None`则以该批次内最长序列长度填充；若`==int`则以该值填充，超过部分截断
    :param batch_first: 是否以batch_size作为返回tensor的第一个维度
    """
    if max_len is None:
        max_len = max([seq.size(0) for seq in sequences])

    out_tensor = []
    # 遍历每个序列，和max_len比较，填充或截断
    for seq in sequences:
        if seq.size(0) < max_len:
            seq = torch.cat(
                [seq, torch.tensor([padding_value] * (max_len - seq.size(0)))],
                dim=0,
            )
        else:
            seq = seq[:max_len]

        out_tensor.append(seq)
    out_tensor = torch.stack(out_tensor, dim=1)
    # 将batch_size作为第一个维度
    if batch_first:
        out_tensor = out_tensor.transpose(0, 1)

    return out_tensor


def cache_decorator(func):
    """
    修饰器——缓存token转换为索引的结果
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs["filepath"]  # 文件路径
        filename = "".join(filepath.split(os.sep)[-1].split(".")[:-1])  # 文件名（不包含拓展名）
        filedir = f"{os.sep}".join(filepath.split(os.sep)[:-1])  # 文件目录

        cache_filename = f"cache_{filename}_token2idx.pt"
        cache_path = os.path.join(filedir, cache_filename)

        start_time = time.time()
        if not os.path.exists(cache_path):
            logging.info(f"缓存文件 {cache_path} 不存在，处理数据集并缓存！")
            data = func(*args, **kwargs)  # token转换为索引
            with open(cache_path, "wb") as f:
                torch.save(data, f)  # 缓存
        else:
            logging.info(f"缓存文件 {cache_path} 存在，载入缓存！")
            with open(cache_path, "rb") as f:
                data = torch.load(f)
        end_time = time.time()

        logging.info(f"数据预处理一共耗时{(end_time - start_time):.3f}s")

        return data

    return wrapper


class LoadSenClsDataset:
    """加载文本分类数据集"""

    def __init__(
        self,
        vocab_path="./vocab.txt",
        tokenizer=None,
        batch_size=32,
        max_sen_len=None,
        split_sep="\n",
        max_position_embeddings=512,
        pad_index=0,
        is_sample_shuffle=True,
    ):
        """
        :param vocab_path: 本地词表路径
        :param tokenizer: 分词器
        :param batch_size: 批次大小
        :param max_sen_len: 填充模式，`="same"`时，按照整个数据集中最长序列填充样本；`=None`时，按照批次内最长序列填充样本；\
            `=int`时，表示以固定长度填充样本，多余的截掉
        :param split_sep: 文本和标签之间的分隔符
        :param max_position_embeddings: 最大序列长度，超过部分将被截断
        :param pad_index: padding token的索引
        :param is_sample_shuffle: 是否打乱数据集，注意仅用于打乱训练集，而不打乱验证集和测试集
        """

        self.vocab = Vocab(vocab_path)  # 读取本地vocab.txt文件，创建词表
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # min(max_sen_len, max_position_embeddings)限制了最大序列长度
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len

        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        self.PAD_IDX = pad_index
        self.CLS_IDX = self.vocab["[CLS]"]
        self.SEP_IDX = self.vocab["[SEP]"]
        self.is_sample_shuffle = is_sample_shuffle

    @cache_decorator
    def token_to_idx(self, filepath=None):
        """
        将token序列转换为索引序列，并返回最长序列长度
        """
        with open(filepath, encoding="utf8") as f:
            raw_iter = f.readlines()
        data = []  # data列表中每个元素表示一个索引序列及标签
        max_len = 0  # 最长序列长度

        for raw_line in tqdm(raw_iter, ncols=80):
            # 取出文本序列和类别标签
            line = raw_line.rstrip("\n").split(self.split_sep)
            text, label = line[0], line[1]

            # 分词、转换为索引序列并添加[CLS]、[SEP] token
            idx_seq = [self.CLS_IDX] + [
                self.vocab[token] for token in self.tokenizer(text)
            ]
            # BERT模型最大支持512个token的序列
            if len(idx_seq) > self.max_position_embeddings - 1:
                idx_seq = idx_seq[: self.max_position_embeddings - 1]
            idx_seq += [self.SEP_IDX]
            idx_seq = torch.tensor(idx_seq, dtype=torch.long)
            label = torch.tensor(int(label), dtype=torch.long)  # 类别标签0~14

            max_len = max(max_len, idx_seq.size(0))
            data.append((idx_seq, label))

        return data, max_len

    def data_loader(
        self,
        train_filepath=None,
        val_filepath=None,
        test_filepath=None,
        only_test=False,
    ):
        """
        创建DataLoader
        :param only_test: 是否只返回测试集
        """
        test_data, _ = self.token_to_idx(filepath=test_filepath)
        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,  # 测试集不打乱
            collate_fn=self.generate_batch,
        )
        if only_test:
            return test_loader

        train_data, max_len = self.token_to_idx(filepath=train_filepath)
        if self.max_sen_len == "same":
            self.max_sen_len = max_len

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=self.is_sample_shuffle,
            collate_fn=self.generate_batch,
        )

        val_data, _ = self.token_to_idx(filepath=val_filepath)
        val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,  # 验证集不打乱
            collate_fn=self.generate_batch,
        )

        return train_loader, test_loader, val_loader

    def generate_batch(self, data_batch):
        """
        对每个批次中的样本进行处理的函数，将作为一个参数传入DataLoader的构造函数
        :param data_batch: 一个批次的数据
        """
        batch_seqs, batch_labels = [], []

        # 遍历一个批次内的样本，取出序列和标签
        for seq, label in data_batch:
            batch_seqs.append(seq)
            batch_labels.append(label)

        batch_seqs = pad_sequence(
            batch_seqs,
            padding_value=self.PAD_IDX,
            max_len=self.max_sen_len,
            batch_first=False,
        )

        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return batch_seqs, batch_labels


class LoadPairSenClsDataset(LoadSenClsDataset):
    """加载文本对分类数据集"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    # 重载子类中的token_to_idx和generate_batch方法
    @cache_decorator
    def token_to_idx(self, filepath=None):
        """
        将token序列转换为索引序列，并返回最长序列长度
        """
        with open(filepath, "r", encoding="utf-8") as f:
            raw_iter = f.readlines()
        data = []  # data列表中每个元素表示一个索引序列、对应的token_type_ids序列及标签
        max_len = 0  # 最长序列长度

        for raw_line in tqdm(raw_iter, ncols=80):
            # 取出两个序列（前提、假设）和类别标签
            line = raw_line.rstrip("\n").split(self.split_sep)
            seq1, seq2, label = line[0], line[1], line[2]
            # 分词并转换为索引序列
            idx_seq1 = [self.vocab[token] for token in self.tokenizer(seq1)]
            idx_seq2 = [self.vocab[token] for token in self.tokenizer(seq2)]
            # 将两个索引序列拼接成一个序列，并添加[CLS]、[SEP] token
            idx_seq = [self.CLS_IDX] + idx_seq1 + [self.SEP_IDX] + idx_seq2

            # BERT模型最大支持512个token的序列
            if len(idx_seq) > self.max_position_embeddings - 1:
                idx_seq = idx_seq[: self.max_position_embeddings - 1]
            idx_seq += [self.SEP_IDX]

            # 创建token_type_id序列，用于表示token所在序列
            seg_seq1 = [0] * (len(idx_seq1) + 2)  # 起始[CLS]和中间的[SEP]两个token属于第一个序列
            seg_seq2 = [1] * (len(idx_seq) - len(seg_seq1))  # 末尾的[SEP]token属于第二个序列

            idx_seq = torch.tensor(idx_seq, dtype=torch.long)
            seg_seq = torch.tensor(seg_seq1 + seg_seq2, dtype=torch.long)
            label = torch.tensor(int(label), dtype=torch.long)  # 类别标签0~2
            max_len = max(max_len, idx_seq.size(0))
            data.append((idx_seq, seg_seq, label))

        return data, max_len

    def generate_batch(self, data_batch):
        """
        对每个批次中的样本进行处理的函数，将作为一个参数传入DataLoader的构造函数
        :param data_batch: 一个批次的数据
        """
        batch_seqs, batch_segs, batch_labels = [], [], []

        # 遍历一个批次内的样本，取出索引序列、token_type_id序列和标签
        for seq, seg, label in data_batch:
            batch_seqs.append(seq)
            batch_segs.append(seg)
            batch_labels.append(label)

        batch_seqs = pad_sequence(
            batch_seqs,
            padding_value=self.PAD_IDX,
            max_len=self.max_sen_len,
            batch_first=False,
        )

        # 对token_type_id序列进行填充，注意：虽然填充id也是0（和第一个序列中的token一样），但是在分类任务中padding token不产生影响
        batch_segs = pad_sequence(
            batch_segs,
            padding_value=self.PAD_IDX,
            max_len=self.max_sen_len,
            batch_first=False,
        )

        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        return batch_seqs, batch_segs, batch_labels
