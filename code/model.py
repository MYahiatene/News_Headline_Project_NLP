import tensorflow as tf
from transformers import TFAutoModel as auto
from preprocessing import data_sets, tokenizer, max_length

data_set = data_sets(tokenizer, max_length)
train_data = data_set.data_set_train
dev_data = data_set.data_set_dev
print(train_data)
print(dev_data)
train_data = train_data.take(1000).shuffle(1000).batch(32, drop_remainder=True)
dev_data = dev_data.take(1000).shuffle(1000).batch(32, drop_remainder=True)
model = auto.from_pretrained("bert-base-cased")
input = tf.keras.layers.Input(shape=(max_length,), name="input", dtype="int32")
mask = tf.keras.layers.Input(shape=(max_length,), name="mask", dtype="int32")
result_emb = model.bert(input, attention_mask=mask)[1]
layer1 = tf.keras.layers.Dense(max_length*2, activation="relu")(result_emb)
layer_output = tf.keras.layers.Dense(3, activation="softmax", name="output_layer")(layer1)
model = tf.keras.Model(inputs=[input, mask], outputs=layer_output)
model.layers[2].trainable=False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy("accuracy")])
model.summary()
history = model.fit(train_data, validation_data=dev_data, epochs=3)
model.save("my_model")
