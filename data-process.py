from tqdm import tqdm
from datasets import load_dataset

# 指定缓存目录为工作区根目录下的 datasets 目录
custom_cache_dir = "./datasets"

# 加载数据集
train_dataset = load_dataset("Skylion007/openwebtext", split="train[:90%]", cache_dir=custom_cache_dir)
val_dataset = load_dataset("Skylion007/openwebtext", split="train[90%:]", cache_dir=custom_cache_dir)

def process_files(files, output_file):
    vocab = set()
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_data in tqdm(files):
            text = file_data["text"]
            outfile.write(text)
            vocab.update(set(text))
    return vocab

# 定义输出文件路径
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"

# 处理训练集文件
train_data = train_dataset
vocab_train = process_files(train_data, output_file_train)

# 处理验证集文件
val_data = val_dataset
vocab_val = process_files(val_data, output_file_val)

# 将词汇表写入文件
vocab_file = "vocab.txt"
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab_train.union(vocab_val)):
        vfile.write(char + '\n')
