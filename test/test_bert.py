import os
import sys

sys.path.append(os.getcwd())
import logging
import torch
from util.log_helper import logger_init
from model.BERT.config import BertConfig
from model.BERT.bert import BertEmbedding
from model.BERT.bert import BertAttention
from model.BERT.bert import BertLayer
from model.BERT.bert import BertEncoder
from model.BERT.bert import BertModel


if __name__ == "__main__":
    logger_init(log_filename="test", log_level=logging.DEBUG)

    json_file = "./archive/bert_base_chinese/config.json"
    config = BertConfig.from_json_file(json_file)
    # # 使用torch框架中的MultiHeadAttention实现
    # config.__dict__["use_torch_multi_head"] = True
    config.max_position_embeddings = 518  # 测试max_position_embeddings大于512时的情况

    # #[src_len, batch_size]
    src = torch.tensor(
        [[1, 3, 5, 7, 9, 2, 3], [2, 4, 6, 8, 10, 0, 0]], dtype=torch.long
    ).transpose(0, 1)
    print(f"input shape #[src_len, batch_size]: ", src.shape)
    # #[src_len, batch_size]
    token_type_ids = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]], dtype=torch.long
    ).transpose(0, 1)

    # attention_mask实质上是padding mask #[src_len, batch_size]
    attention_mask = torch.tensor(
        [
            [False, False, False, False, False, True, True],
            [False, False, False, False, False, False, True],
        ]
    )

    print("------ 测试BertEmbedding -------")
    bert_embedding = BertEmbedding(config)
    bert_embed_out = bert_embedding(src, token_type_ids=token_type_ids)
    print(
        f"BertEmbedding output shape #[src_len, batch_size, hidden_size]: {bert_embed_out.shape}"
    )

    print("------ 测试BertAttention -------")
    bert_attention = BertAttention(config)
    bert_attn_out = bert_attention(bert_embed_out, attention_mask=attention_mask)
    print(
        f"BertAttention output shape #[src_len, batch_size, hidden_size]: {bert_attn_out.shape}",
    )

    print("------ 测试BertLayer -------")
    bert_layer = BertLayer(config)
    bert_layer_out = bert_layer(bert_embed_out, attention_mask)
    print(
        f"BertLayer output shape #[src_len, batch_size, hidden_size]: {bert_layer_out.shape}",
    )

    print("------ 测试BertEncoder -------")
    bert_encoder = BertEncoder(config)
    bert_encoder_outs = bert_encoder(bert_embed_out, attention_mask)
    print(
        f"Num of BertEncoder outputs #[config.num_hidden_layers]: {len(bert_encoder_outs)}",
    )
    print(
        f"Each output shape of BertEncoder #[src_len, batch_size, hidden_size]: {bert_encoder_outs[0].shape}",
    )

    print("------ 测试BertModel -------")
    position_ids = torch.arange(src.size()[0]).expand((1, -1))  # [1, src_len]
    bert_model = BertModel(config)
    bert_pooler_out = bert_model(
        input_ids=src,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )[0]
    print(
        f"BertPooler output shape #[batch_size, hidden_size]: {bert_pooler_out.shape}",
    )
    print("======= BertModel参数: ========")
    for param in bert_model.state_dict():
        print(param, "\t #", bert_model.state_dict()[param].size())

    print("------ 测试BertModel载入预训练模型 -------")
    model = BertModel.from_pretrained(
        config=config, pretrained_model_dir="./archive/bert_base_chinese"
    )
