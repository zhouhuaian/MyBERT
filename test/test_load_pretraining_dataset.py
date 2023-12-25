import os
import sys

sys.path.append(os.getcwd())

from tasks.pretraining import ModelConfig
from utils import LoadPretrainingDataset
from transformers import BertTokenizer


if __name__ == "__main__":
    model_config = ModelConfig()
    dataset = LoadPretrainingDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(
            model_config.pretrained_model_dir
        ).tokenize,
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        max_position_embeddings=model_config.max_position_embeddings,
        split_sep=model_config.split_sep,
        pad_index=model_config.pad_index,
        is_sample_shuffle=model_config.is_sample_shuffle,
        dataset_name=model_config.dataset_name,
        masked_rate=model_config.masked_rate,
        masked_token_rate=model_config.masked_token_rate,
        masked_token_unchanged_rate=model_config.masked_token_unchanged_rate,
        random_state=model_config.random_state,
    )

    train_loader, val_loader, test_loader = dataset.data_loader(
        train_filepath=model_config.train_filepath,
        val_filepath=model_config.val_filepath,
        test_filepath=model_config.test_filepath,
    )
    # # 仅生成测试集
    # test_loader = dataset.data_loader(
    #     test_filepath=model_config.test_filepath, only_test=True
    # )

    for id_seqs, segs, b_mask, mlm_labels, nsp_labels in test_loader:
        print(f"token id seqs shape: #{id_seqs.shape}")  # [src_len, batch_size]
        print(f"token type id seqs shape: #{segs.shape}")  # [src_len, batch_size]
        print(f"padding mask shape: #{b_mask.shape}")  # [batch_size, src_len]
        print(f"MLM task labels shape: #{mlm_labels.shape}")  # [src_len, batch_size]
        print(f"NSP task labels shape: #{nsp_labels.shape}")  # [batch_size]

        id_seq = id_seqs.transpose(0, 1)[0]
        mlm_label = mlm_labels.transpose(0, 1)[0]
        token = " ".join([dataset.vocab.itos[id] for id in id_seq])
        label = " ".join([dataset.vocab.itos[id] for id in mlm_label])
        print(f"Mask后的token序列：{token}")
        print(f"对应的MLM任务标签：{label}")

        break

    sentences = ["十年生死两茫茫。不思量。自难忘。千里孤坟，无处话凄凉。", "红酥手。黄藤酒。满园春色宫墙柳。"]
    b_id_seq, b_mask_pos, b_padding_mask = dataset.get_inference_samples(
        sentences, masked=False
    )
    print("=" * 10, "推理时示例样本", "=" * 10)
    print(f"token id seqs: {b_id_seq.transpose(0, 1)}")
    print(f"MLM task labels: {b_mask_pos}")
    print(f"padding mask: {b_padding_mask}")
