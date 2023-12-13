import os
import sys

sys.path.append(os.getcwd())
import torch
from model.downstream.bert_for_sen_cls import BertForSenCls
from model.BERT.config import BertConfig

if __name__ == "__main__":
    json_file = "./archive/bert_base_chinese/config.json"
    config = BertConfig.from_json_file(json_file)
    config.__dict__["num_labels"] = 10
    # # config.__dict__["num_hidden_layers"] = 3
    model = BertForSenCls(config)

    # #[src_len, batch_size] [6, 2]
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.long
    ).transpose(0, 1)
    # #[batch_size, src_len] [2, 6]
    attention_mask = torch.tensor(
        [
            [False, False, False, False, False, True],
            [False, False, False, True, True, True],
        ]
    )
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    print(logits.shape)  # #[batch_size, num_labels] [2, 10]
