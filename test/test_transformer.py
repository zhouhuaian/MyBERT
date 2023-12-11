import os
import sys

sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from model.BERT.transformer import MultiheadAttention
from model.BERT.transformer import EncoderLayer, Encoder
from model.BERT.transformer import DecoderLayer, Decoder
from model.BERT.transformer import Transformer


if __name__ == "__main__":
    batch_size = 2
    src_len = 5
    tgt_len = 7
    d_model = 32
    nhead = 4

    src = torch.rand((src_len, batch_size, d_model))  # [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor(
        [[False, False, False, True, True], [False, False, False, False, True]]
    )  # [batch_size, src_len]

    tgt = torch.rand((tgt_len, batch_size, d_model))  # [tgt_len, batch_size, embed_dim]
    tgt_key_padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False, True],
            [False, False, False, False, True, True, True],
        ]
    )  # [batch_size, tgt_len]

    print("============ 测试 MultiheadAttention ============")
    my_mh = MultiheadAttention(embed_dim=d_model, num_heads=nhead)
    my_mh_out = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)
    print(my_mh_out[0].shape)  # [5, 2, 32]

    mh = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
    mh_out = mh(src, src, src, key_padding_mask=src_key_padding_mask)
    print(mh_out[0].shape)  # [5, 2, 32]

    print("============ 测试 Encoder ============")
    enlayer = EncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512)
    encoder = Encoder(enlayer, num_layers=4, norm=nn.LayerNorm(d_model))
    memory = encoder(src, src_key_padding_mask=src_key_padding_mask)
    print(memory.shape)  # [5, 2, 32]

    print("============ 测试 Decoder ============")
    delayer = DecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256)
    decoder = Decoder(delayer, num_layers=6, norm=nn.LayerNorm(d_model))
    out = decoder(
        tgt,
        memory,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )
    print(out.shape)  # [7, 2, 32]

    print("============ 测试 Transformer ============")
    my_transformer = Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=256,
    )
    tgt_mask = my_transformer.generate_attn_mask(tgt_len)
    output = my_transformer(
        src=src,
        tgt=tgt,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )
    print(output.shape)  # [7, 2, 32]
