import os
import sys

sys.path.append(os.getcwd())

from tasks.pair_sentence_classification import ModelConfig
from utils.data_helpers import LoadPairSenClsDataset
from transformers import BertTokenizer

if __name__ == "__main__":
    model_config = ModelConfig()
    dataset = LoadPairSenClsDataset(
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

    train_loader, val_loader, test_loader = dataset.data_loader(
        model_config.train_filepath,
        model_config.val_filepath,
        model_config.test_filepath,
    )

    for seqs, segs, labels in train_loader:
        print(seqs.shape)  # #[seq_len, batch_size]
        print(seqs.transpose(0, 1))  # #[batch_size, seq_len]
        print(segs.shape)  # #[seq_len, batch_size]
        print(labels.shape)  # #[batch_size,]
        print(labels)

        padding_mask = (seqs == dataset.PAD_IDX).transpose(0, 1)
        print(padding_mask.shape)  # #[batch_size, seq_len]

        break
