'''import transformers
import tensorflow as tf
from transformers import AutoModelForSequenceClassification
from preprocessing import prepare_data
train1, train2, test1, test2, dev1, dev2 =prepare_data()
print(train1.__getitem__(0))
model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-cased',num_labels=2)


'''