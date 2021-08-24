import tensorflow
from model import build_model, test_data

if __name__ == "__main__":
    '''model = build_model()
    model.summary()
    history = model.fit(train_data, validation_data=dev_data, epochs=3)'''
    model = tensorflow.keras.models.load_model("my_model")
    # model.evaluate(test_data, return_dict=True)
