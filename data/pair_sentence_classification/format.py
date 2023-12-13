import json
import numpy as np

label_map = {"contradiction": "0", "entailment": "1", "neutral": "2"}


def format(filepath="./multinli_1.0_train.jsonl"):
    np.random.seed(42)
    with open(filepath, "r", encoding="utf-8") as f:
        raw_data = f.readlines()

    num_samples = len(raw_data)
    num_train, num_val = int(0.7 * num_samples), int(0.2 * num_samples)
    num_test = num_samples - num_train - num_val
    # 生成随机索引及三个数据集样本索引
    idx = np.random.permutation(num_samples)
    train_idx, val_idx, test_idx = (
        idx[:num_train],
        idx[num_train : num_train + num_val],
        idx[-num_test:],
    )
    #  写入文件
    f_train = open("./train.txt", "w", encoding="utf-8")
    f_val = open("./val.txt", "w", encoding="utf-8")
    f_test = open("./test.txt", "w", encoding="utf-8")

    for i in train_idx:
        line = raw_data[i].strip("\n")
        sample = json.loads(line)
        tmp = (
            sample["sentence1"]
            + "_!_"
            + sample["sentence2"]
            + "_!_"
            + label_map[sample["gold_label"]]
        )
        f_train.write(tmp + "\n")
    f_train.close()

    for i in val_idx:
        line = raw_data[i].strip("\n")
        sample = json.loads(line)
        tmp = (
            sample["sentence1"]
            + "_!_"
            + sample["sentence2"]
            + "_!_"
            + label_map[sample["gold_label"]]
        )
        f_val.write(tmp + "\n")
    f_val.close()

    for i in test_idx:
        line = raw_data[i].strip("\n")
        sample = json.loads(line)
        tmp = (
            sample["sentence1"]
            + "_!_"
            + sample["sentence2"]
            + "_!_"
            + label_map[sample["gold_label"]]
        )
        f_test.write(tmp + "\n")
    f_test.close()


if __name__ == "__main__":
    # format(filepath="./multinli_1.0_train.jsonl")
    format()
