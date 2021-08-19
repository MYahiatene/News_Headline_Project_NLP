import tensorflow as tf
from transformers import TFAutoModel
from preprocessing import prepare_data, max_length

train_data, dev_data = prepare_data()
train_data=train_data.shuffle(100).batch(128, drop_remainder=True)
dev_data= dev_data.shuffle(100).batch(128, drop_remainder=True)
model = TFAutoModel.from_pretrained("bert-base-cased")
input = tf.keras.layers.Input(shape=(max_length,), name="input", dtype="int32")
mask = tf.keras.layers.Input(shape=(max_length,), name="mask", dtype="int32")
result_emb = model.bert(input, attention_mask=mask)[1]
layer1 = tf.keras.layers.Dense(max_length * 2, activation="relu")(result_emb)
layer_output = tf.keras.layers.Dense(3, activation="softmax", name="output_layer")(layer1)
model = tf.keras.Model(inputs=[input, mask], outputs=layer_output)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5, decay=1e-6), loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy("accuracy")])
model.summary()
model.layers[2].trainable=False
history = model.fit(train_data,validation_data=dev_data, epochs=1)
