import transformers
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from preprocessing import prepare_data
input_ids_train, attention_masks_train, input_ids_test, attention_masks_test, input_ids_dev, attention_masks_dev =prepare_data()
print(input_ids_train)
#model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-cased',num_labels=2)
