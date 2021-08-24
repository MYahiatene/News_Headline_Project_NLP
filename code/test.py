import transformers
import tensorflow
import numpy as np
from preprocessing import max_length

model = tensorflow.keras.models.load_model("my_model")


def predict(original, edit1, edit2):
    original = original
    edit1 = edit1
    edit2 = edit2
    text = original + edit1 + original + edit2
    tokens = transformers.BertTokenizer.from_pretrained("bert-base-cased").encode_plus(text=text, max_length=max_length,
                                                                                       truncation=True,
                                                                                       padding='max_length',
                                                                                       add_special_tokens=True,
                                                                                       return_token_type_ids=False,
                                                                                       return_tensors='tf')
    prediction_softmax = model.predict(dict(input=tokens["input_ids"], mask=tokens["attention_mask"]))

    print(prediction_softmax)
    prediction = np.argmax(prediction_softmax)
    print("Edit{} is funnier! :)".format(prediction))
    return np.argmax(prediction_softmax)


original = "40 percent of voters believe Trump is fit to be president , a new low "
edit1 = "40 percent of gnomes believe Trump is fit to be president , a new low "
edit2 = "40 percent of voters believe Trump is fit to be triathlete , a new low "
predict(original, edit1, edit2)
