import json
from tqdm import tqdm


def format():
    """划分原始的ci.song.xxx.json数据集，保存为训练、验证和测试集"""

    def read_file(pathname=None):
        """读取每个json文件中的1000首词"""

        paragraphs = []
        with open(pathname, encoding="utf-8") as f:
            json_data = json.loads(f.read())
            for item in json_data:
                para = item["paragraphs"]  # 取出一首词（一个段落）的列表
                # 去除"词牌介绍"和" >> "两句
                if para[-1] == "词牌介绍":
                    para = para[:-2]
                if len(para) < 2:  # 舍弃小于两句的词
                    continue
                paragraphs.append(para)

        return paragraphs

    def make_data(pathname, start, end):
        with open(pathname, "w", encoding="utf-8") as f:
            for i in tqdm(
                range(start, end, 1000), ncols=80, desc=f"## 正在制作数据集{pathname}"
            ):
                json_file = f"ci.song.{i}.json"
                paragraphs = read_file(json_file)
                for para in paragraphs:
                    f.write("".join(para) + "\n")

    make_data("songci.train.txt", 0, 19001)  # 20 * 1000首
    make_data("songci.valid.txt", 20000, 21001)  # 2 * 1000首
    make_data("songci.test.txt", 20000, 21001)  # 2 * 1000首


if __name__ == "__main__":
    format()
