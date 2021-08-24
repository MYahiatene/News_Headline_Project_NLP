
import tensorflow as tf

if __name__ == "__main__":
    model = tf.keras.models.load_model("./my_model")
    original = "40 percent of voters believe Trump is fit to be president , a new low "
    edit1 = "40 percent of gnomes believe Trump is fit to be president , a new low "
    edit2 = "40 percent of voters believe Trump is fit to be triathlete , a new low "
    #test= pd.read_csv("test.csv")
    #test_input,test_mask,_=data_sets._prepare_(test)
    #for i,idx in enumerate(test_input):
     #model.predict(dict(input=tokens["input_ids"], mask=tokens["attention_mask"])
"""    predictions = list()
    text = original + edit1 + original + edit2
    tokens = transformers.BertTokenizer.from_pretrained("bert-base-cased").encode_plus(text=text, max_length=128,
                                                                                       truncation=True,
                                                                                       padding='max_length',
                                                                                       add_special_tokens=True,
                                                                                       return_token_type_ids=False,
                                                                                       return_tensors='tf')
    test = model.predict(dict(input=tokens["input_ids"], mask=tokens["attention_mask"]))

    print(test)
    print(np.argmax(test))"""
    #print(predictions)
