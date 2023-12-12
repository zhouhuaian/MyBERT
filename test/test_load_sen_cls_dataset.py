import os
import sys

sys.path.append(os.getcwd())

from tasks.sentence_classification import ModelConfig
from utils.data_helpers import LoadSenClsDataset
from transformers import BertTokenizer  # 借用transformers框架中的分词器

# 加载数据集和预处理：1.分词（使用BertTokenizer）；2.创建词表（读取已有文件vocab.txt）；3.把token转换为索引序列，添加[CLS]和[SEP] token；
# 4.填充；5.构造DataLoader

if __name__ == "__main__":
    model_config = ModelConfig()
    dataset = LoadSenClsDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(
            model_config.pretrained_model_dir
        ).tokenize,
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle,
    )

    train_loader, test_loader, val_loader = dataset.data_loader(
        model_config.train_filepath,
        model_config.val_filepath,
        model_config.test_filepath,
    )

    for sample, label in train_loader:
        print(sample.shape)  # #[seq_len, batch_size]
        print(sample.transpose(0, 1))
        print(label)

        # #[batch_size, seq_len]
        padding_mask = (sample == dataset.PAD_IDX).transpose(0, 1)
        print(padding_mask)

        break
