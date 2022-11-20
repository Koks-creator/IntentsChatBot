import pickle
import json
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd

from ChatBot.chatbotPipelines.Pipelines import setup_preprocessing_pipeline

with open("../intents4.json") as f:
    intents = json.load(f)

X = []
y = []

for intent in intents["intents"]:
    for pattern in intent["text"]:
        X.append(pattern)
        y.append(intent["intent"])

df = pd.DataFrame(
    {
        "X": X,
        "y": y
    }
)

preprocessing_pipeline = setup_preprocessing_pipeline(
    x_column_name="X",
    labels_column="y",
    max_length=10,
    label_encoder_path=r"data/label_encoder.pkl",
    tokenizer_path=r"data/tokenizer.pkl",
    unique_words_path=r"data/words.pkl",
    data_output_path=r"data/train_data.pkl",

)

data = preprocessing_pipeline.fit_transform(df)

with open("data/words.pkl", "rb") as f:
    words = pickle.load(f)


# with open("data.pkl", "rb") as f:
#     data = pickle.load(f)

X_train = data["X"]
y_train = data["y"]

y = y_train.copy()
classes_list = list(y)
number_of_classes = len(set(map(tuple, classes_list)))

model = keras.models.Sequential()

model.add(layers.Embedding(len(words), 32, input_length=10))
model.add(layers.LSTM(64, dropout=0.1))
model.add(layers.Dense(72, activation="softmax"))

model.summary()

optim = keras.optimizers.Adam(lr=0.001)

model.compile(
    optimizer=optim,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=100, verbose=1)

model.save("Test.h5")