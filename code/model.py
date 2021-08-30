import tensorflow as tf
from transformers import TFBertModel as auto
from preprocessing import data_sets, tokenizer, max_length, batch_size

data_set = data_sets(tokenizer,
                     max_length)  # creates our data set object with a specified tokenizer and max sequence length

train_data = data_set.data_set_train  # train data
dev_data = data_set.data_set_dev  # validation data
test_data = data_set.data_set_test  # test data

train_data = train_data.shuffle(1000).batch(batch_size, drop_remainder=True)  # shuffle and batch train data
dev_data = dev_data.shuffle(1000).batch(batch_size, drop_remainder=True)  # shuffle and batch validation data
test_data = test_data.batch(batch_size)  # batch test data


# function to build and return our model
def build_model():
    '''
    :return: Fine tuned bert model
    :rtype: tensorflow model
    '''
    model = auto.from_pretrained("bert-base-cased")
    input = tf.keras.layers.Input(shape=(max_length,), name="input", dtype="int32")
    mask = tf.keras.layers.Input(shape=(max_length,), name="mask", dtype="int32")
    result_emb = model.bert(input, attention_mask=mask)[1]
    layer1 = tf.keras.layers.Dense(1024, activation="relu")(result_emb)
    layer_output = tf.keras.layers.Dense(3, activation="softmax", name="output_layer")(layer1)
    model = tf.keras.Model(inputs=[input, mask], outputs=layer_output)
    # model.layers[2].trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-8),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy("accuracy")])
    return model
