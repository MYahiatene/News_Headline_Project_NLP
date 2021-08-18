import pandas as pd
import transformers
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

# id,        original1,                                                       edit1,grades1,meanGrade1,original2,                                                edit2,      grades2,meanGrade2,label
# 10920-9866,""" Gene Cernan , Last <Astronaut/> on the Moon , Dies at 82 """,Dancer,01113,1.2,""" Gene Cernan , Last Astronaut on the Moon , <Dies/> at 82 """,impregnated,30001,0.8,1

labels = [0, 1, 2, 3]


def import_corpus():
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    valid = pd.read_csv("dev.csv")
    return train, test, valid


class data_set():
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_len: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx: int):
        text_edited = self.data.iloc[idx]["original"].split("<")[0] + self.data.iloc[idx]["edit"] + \
                      self.data.iloc[idx]["original"].split(">")[1]
        meanGrade = self.data.iloc[idx]["meanGrade"]
        encoding = self.tokenizer.encode_plus(text_edited, add_special_tokens=True, max_length=self.max_len,
                                              return_token_type_ids=False, padding="max_length", truncation=True,
                                              return_attention_mask=True, return_tensors="tf")
        return dict(text=text_edited, input_ids=tf.squeeze(encoding["input_ids"]),
                    attention_mask=tf.squeeze(encoding["attention_mask"]),meanGrade=meanGrade)


bert_model = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(bert_model)
train, test, dev = import_corpus()
complete_data = train.append(test, ignore_index=True)
complete_data = complete_data.append(dev, ignore_index=True)
data_ver1 = complete_data[["original1", "edit1", "meanGrade1"]]
data_ver2 = complete_data[["original2", "edit2", "meanGrade2"]]
data_ver1.rename(columns={"original1": "original", "edit1": "edit", "meanGrade1": "meanGrade"},
                 inplace=True)
data_ver2.rename(columns={"original2": "original", "edit2": "edit", "meanGrade2": "meanGrade"},
                 inplace=True)
data_train = data_set(data_ver1, tokenizer)
print(data_train.__getitem__(0))

'''tokenizer=BertTokenizer.from_pretrained(bert_model)
print(train["original2"][0])
print(train["edit2"][0])
print(tokenizer.encode_plus(train["original2"][0],max_length=130,padding="max_length"))
print(tokenizer.decode(tokenizer(train["original1"][0],train["original2"][0])["input_ids"]))
print(tokenizer(train["original1"][0],train["original2"][0],return_tensors="tf"))'''
