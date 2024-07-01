import json
import pickle
import re
from difflib import SequenceMatcher
import os

import pandas as pd


def preprocess_text(text):
    """
    文本预处理,调整格式,去除掉一些符号的影响
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.replace('\n', '').replace('\r', '')
    return text


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def load_json_data(json_data, preprocessed_content):
    """
    模糊查找
    json_data:json数据,style_based_fake_news.json
    preprocessed_content:预处理之后的pkl中的数据像本的content文本内容
    """
    max_similarity = 0
    max_key = None
    max_value = None
    for key, value in json_data.items():
        preprocessed_origin_text = preprocess_text(value['origin_text'])
        sim = similarity(preprocessed_content, preprocessed_origin_text)
        if sim > max_similarity:
            max_similarity = sim
            max_key = key
            max_value = value
    if max_similarity > 0.5:
        # 相似度最大的那条记录的相似度大于50%,即为查找成功
        return max_key, max_value
    return None, None


def create_new_pkl(pkl_path,json_data):
    """
    逐条查找,边差边改边删:查找成功则修改pkl样本,删除json数据项;查找失败则删除pkl样本
    pkl_path:需要进行处理的pkl文件路径
    json_data: ,style_based_fake_news.json
    """
    pkl_name = os.path.splitext(os.path.basename(pkl_path))[0]
    with open(pkl_path, 'rb') as file:
        pkl_data = pickle.load(file)
        # 获取所有属性
        attributes = list(pkl_data.keys())

        # 存储查找成功的记录
        found = []

        # 逐条查找,边差边改边删
        # 查找成功则修改pkl样本,删除json数据项
        # 查找失败则删除pkl样本
        for i, news_content in enumerate(pkl_data["content"]):
            preprocessed_content = preprocess_text(news_content)
            if pkl_data["category"][i] == "gossipcop":
                # 去json文件中进行模糊查找
                key, value = load_json_data(json_data, preprocessed_content)
                if key is not None:
                    print(f"Replaced news content for news {i} with generated_text from {key}")
                    # 修改查找成功的数据样本
                    pkl_data["content"][i] = value['generated_text']
                    # 删除对应的json数据项,加速后续查找速度
                    del json_data[key]
                    # 添加查找成功的记录
                    found.append((i, key))
                else:
                    print(f"No matching key found for news {i}")
                    # 删除查找失败项
                    for attr in attributes:
                        if isinstance(pkl_data[attr], (pd.Series, pd.DataFrame)):
                            if i in pkl_data[attr].index:
                                pkl_data[attr] = pkl_data[attr].drop(i)
                        elif isinstance(pkl_data[attr], list) and i < len(pkl_data[attr]):
                            del pkl_data[attr][i]

    # 打印查找成功的数量
    print(f"{pkl_name} : length of replacements is {len(found)}.")
    # 统计真假新闻数量
    fake_news_count = sum(1 for label in pkl_data["label"] if label == 0)
    real_news_count = sum(1 for label in pkl_data["label"] if label == 1)
    print(f"The number of fake news in {pkl_name} is {fake_news_count}.")
    print(f"The number of real news in {pkl_name} is {real_news_count}.")

    # 保存found列表为json文件
    with open(f'./{pkl_name}_found.json', 'w') as found_file:
        json.dump(found, found_file)

    # 将修改后的数据保存回pkl文件
    with open(pkl_path, 'wb') as file:
        pickle.dump(pkl_data, file)


v3_1_path="../mldata/gossipcop_v3-1_style_based_fake_new.json"
train_pkl_path="../en-v3-1-00/train.pkl"
test_pkl_path="../en-v3-1-00/test.pkl"
val_pkl_path="../en-v3-1-00/val.pkl"

# 一次性加载整个 JSON 文件
with open(v3_1_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 修改val.pkl
create_new_pkl(val_pkl_path,json_data)
# 修改test.pkl
create_new_pkl(test_pkl_path,json_data)
# 修改train.pkl
create_new_pkl(train_pkl_path,json_data)


