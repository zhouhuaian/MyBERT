import os
import sys

sys.path.append(os.getcwd())
from model.BERT.config import BertConfig

if __name__ == "__main__":
    json_file = "./archive/bert_base_chinese/config.json"
    config = BertConfig.from_json_file(json_file)

    for key, value in config.__dict__.items():
        print(f"{key} = {value}")

    print("=" * 20)
    print(config.to_json_str())
