import os
import sys

sys.path.append(os.getcwd())
import torch
from model.BERT.config import BertConfig
from model.BERT.embedding import TokenEmbedding
from model.BERT.embedding import PositionalEmbedding
from model.BERT.embedding import SegmentEmbedding
from model.BERT.embedding import BertEmbedding


if __name__ == "__main__":
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
