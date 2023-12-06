import torch
import torch.nn as nn
from torch.nn.init import normal_
from bert_config import BertConfig


class TokenEmbedding(nn.Module):
    """词嵌入层"""

    def __init__(self, vocab_size, hidden_size, initializer_range=0.02):
        """
        Args:
            vocab_size: 词表大小.
            hidden_size: 词嵌入维度.
            initializer_range: 初始化参数的采样方差.
        """
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, input_ids):
        """
        Args:
            input_ids: `#[token_ids_len, batch_size]`
        Return:
            `#[token_ids_len, batch_size, hidden_size]`
        """
        return self.embedding(input_ids)

    def _reset_parameters(self, initializer_range):
        """初始化参数"""
        for param in self.parameters():
            if param.dim() > 1:
                normal_(param, mean=0.0, std=initializer_range)


class PositionalEmbedding(nn.Module):
    """
    位置编码层——表明token位于序列中的哪个位置\ 
    注意：和Transformer中使用公式编码位置不同，BERT中的位置编码本质上就是普通的Embedding层，
    所以有最大表长（一般设置为512）
    """

    def __init__(
        self, hidden_size, max_position_embeddings=512, initializer_range=0.02
    ):
        """
        Args:
            hidden_size: 词嵌入维度.
            max_position_embeddings: 最大位置，决定了输入模型的文本序列长度.
            initializer_range: 初始化参数的采样方差.
        """
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, position_ids):
        """
        Args:
            position_ids: `#[1, position_ids_len]`
        Return:
            `#[position_ids_len, 1, hidden_size]`
        """
        return self.embedding(position_ids).transpose(0, 1)

    def _reset_parameters(self, initializer_range):
        """初始化参数"""
        for param in self.parameters():
            if param.dim() > 1:
                normal_(param, mean=0.0, std=initializer_range)


class SegmentEmbedding(nn.Module):
    """
    段编码层——表明token位于哪一个序列（段落）
    """

    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        """
        Args:
            type_vocab_size: 序列（段落）数目.
            hidden_size: 词嵌入维度.
            initializer_range: 初始化参数的采样方差.
        """
        super(SegmentEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, hidden_size)
        self._reset_parameters(initializer_range)

    def forward(self, token_type_ids):
        """
        Args:
            token_type_ids: `#[token_type_ids_len, batch_size]`
        Return:
            `#[token_type_ids_len, batch_size, hidden_size]`
        """
        return self.embedding(token_type_ids)

    def _reset_parameters(self, initializer_range):
        """初始化参数"""
        for param in self.parameters():
            if param.dim() > 1:
                normal_(param, mean=0.0, std=initializer_range)


class BertEmbedding(nn.Module):
    """
    BertModel的Input Embedding层\ 
    由Token、Positional、Segment三个Embedding相加得到
    """

    def __init__(self, config: BertConfig):
        super().__init__()  # Python3中可直接写super().__init__()
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
        )
        self.position_embedding = PositionalEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
        )
        self.segment_embedding = SegmentEmbedding(
            type_vocab_size=config.type_vocab_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
        )

        self.layernorm = nn.LayerNorm(config.hidden_size)  # 层标准化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 提前创建所有position id #[1, max_position_embeddings]
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None):
        """
        Args:
            input_ids: 输入序列的原始token id `#[src_len, batch_size]`.
            position_ids: token的位置id，即[0, 1, 2, ..., src_len-1] `#[1, src_len]`.
            token_type_ids: token所在序列id，例如[0, 0, 0, 0, 1, 1, 1]用于区分两个句子 `#[src_len, batch_size]`.
        Return:
            `#[src_len, batch_size, hidden_size]`
        """
        src_len = input_ids.size(0)
        # #[src_len, batch_size, hidden_size]
        token_embed = self.token_embedding(input_ids)

        if position_ids is None:  # 实际上无需传入此参数，取出前面缓存的positional ids的前src_len个即可
            position_ids = self.position_ids[:, :src_len]  # #[1, src_len]
        # #[src_len, 1, hidden_size]
        pos_embed = self.position_embedding(position_ids)

        if token_type_ids is None:  # 若输入模型的仅有一个序列，则无需传入该参数
            token_type_ids = torch.zeros_like(
                input_ids, device=self.position_ids.device
            )  # #[src_len, batch_size]
        # #[src_len, batch_size, hidden_size]
        seg_embed = self.segment_embedding(token_type_ids)

        # 相加
        input_embed = token_embed + pos_embed + seg_embed
        input_embed = self.layernorm(input_embed)
        input_embed = self.dropout(input_embed)

        return input_embed


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())

    json_file = "./archive/bert_base_chinese/config.json"
    config = BertConfig.from_json_file(json_file)

    src = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
    src = src.transpose(0, 1)  # #[src_len, batch_size] [5, 2]

    print("***** --------- 测试TokenEmbedding ------------")
    token_embedding = TokenEmbedding(vocab_size=16, hidden_size=32)
    token_embed = token_embedding(input_ids=src)
    print("src shape #[src_len, batch_size]: ", src.shape)
    print(
        f"token embedding shape #[src_len, batch_size, hidden_size]: {token_embed.shape}\n"
    )

    print("***** --------- 测试PositionalEmbedding ------------")
    # #[1, src_len]
    position_ids = torch.arange(src.shape[0]).expand((1, -1))
    position_embedding = PositionalEmbedding(max_position_embeddings=8, hidden_size=32)
    pos_embed = position_embedding(position_ids=position_ids)
    # print(position_embedding.embedding.weight)  # embedding 矩阵
    print("position_ids shape #[1, src_len]: ", position_ids.shape)
    print(f"positional embedding shape #[src_len, 1, hidden_size]: {pos_embed.shape}\n")

    print("***** --------- 测试SegmentEmbedding ------------")
    token_type_ids = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.long
    ).transpose(0, 1)
    segmet_embedding = SegmentEmbedding(type_vocab_size=2, hidden_size=32)
    seg_embed = segmet_embedding(token_type_ids)
    print("token_type_ids shape #[src_len, batch_size]: ", token_type_ids.shape)
    print(
        f"segment embedding shape #[src_len, batch_size, hidden_size]: {seg_embed.shape}\n"
    )

    print("***** --------- 测试BertEmbedding ------------")
    bert_embedding = BertEmbedding(config)
    input_embed = bert_embedding(src, token_type_ids=token_type_ids)
    print(
        f"input embedding shape #[src_len, batch_size, hidden_size]: {input_embed.shape}"
    )
