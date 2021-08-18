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
    dev = pd.read_csv("dev.csv")
    train_data_ver1 = train[["original1", "edit1", "meanGrade1"]]
    train_data_ver2 = train[["original2", "edit2", "meanGrade2"]]
    train_data_ver1 = train_data_ver1.rename(columns={"original1": "original", "edit1": "edit", "meanGrade1": "meanGrade"})
    train_data_ver2 = train_data_ver2.rename(columns={"original2": "original", "edit2": "edit", "meanGrade2": "meanGrade"})
    test_data_ver1 = test[["original1", "edit1", "meanGrade1"]]
    test_data_ver2 = test[["original2", "edit2", "meanGrade2"]]
    test_data_ver1 = test_data_ver1.rename(columns={"original1": "original", "edit1": "edit", "meanGrade1": "meanGrade"})
    test_data_ver2 = test_data_ver2.rename(columns={"original2": "original", "edit2": "edit", "meanGrade2": "meanGrade"})
    dev_data_ver1 = dev[["original1", "edit1", "meanGrade1"]]
    dev_data_ver2 = dev[["original2", "edit2", "meanGrade2"]]
    dev_data_ver1 = dev_data_ver1.rename(columns={"original1": "original", "edit1": "edit", "meanGrade1": "meanGrade"})
    dev_data_ver2 = dev_data_ver2.rename(columns={"original2": "original", "edit2": "edit", "meanGrade2": "meanGrade"})
    return (train_data_ver1,train_data_ver2), (test_data_ver1,test_data_ver2),(dev_data_ver1,dev_data_ver2)


class data_set():
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_len: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        text_original= row["original"].replace("<", "").replace("/>","")
        text_edited = row["original"].split("<")[0] + row["edit"] + \
                      row["original"].split(">")[1]
        meanGrade = row["meanGrade"]
        encoding = self.tokenizer.encode_plus(text_original,text_edited, add_special_tokens=True, max_length=self.max_len,
                                              return_token_type_ids=False, padding="max_length", truncation=True,
                                              return_attention_mask=True, return_tensors="tf")
        return dict(text_edited=text_edited,text_original=text_original, input_ids=tf.squeeze(encoding["input_ids"]),
                    attention_mask=tf.squeeze(encoding["attention_mask"]), meanGrade=meanGrade)


bert_model = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(bert_model)
(train1,train2), (test1,test2),(dev1,dev2) = import_corpus()
train1=data_set(train1,tokenizer)
train2=data_set(train2,tokenizer)
dict1 = train1.__getitem__(0)
dict2 = train2.__getitem__(0)
print(tokenizer.decode(dict2["input_ids"]))

