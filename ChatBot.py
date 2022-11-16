from dataclasses import dataclass
import string
import random
import pickle
import nltk
import json
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')


@dataclass()
class PreprocessSents:
    tokenizer_file_path: str
    label_encoder_file_path: str
    intents_json_file_path: str

    def __post_init__(self):
        with open(self.tokenizer_file_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(self.label_encoder_file_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(self.intents_json_file_path) as intents_file:
            self.intents = json.loads(intents_file.read())

        self.lemmitizer = WordNetLemmatizer()

    @staticmethod
    def remove_punct(word: str) -> str:
        translator = str.maketrans("", "", string.punctuation)

        return word.translate(translator)

    def lemmatize_string(self, raw_text: str) -> str:
        filtered_text = [self.lemmitizer.lemmatize(word.lower()) for word in raw_text.split()]
        return " ".join(filtered_text)

    def tokenize_sent(self, sent: str) -> list:
        return self.tokenizer.texts_to_sequences(np.array([sent]))

    def pad_sent(self, tokenized_sent: list) -> np.array:
        return pad_sequences(tokenized_sent, maxlen=5, padding="post", truncating="post")

    def decode_class(self, class_id: np.array) -> str:
        return self.label_encoder.inverse_transform([class_id])

    def get_answer(self, pred: np.array) -> str:
        for intent in self.intents["intents"]:
            tag = intent["tag"]
            if tag == pred[0]:
                return random.choice(intent["responses"])

    def full_preprocessing(self, sent: str) -> np.array:
        sent = self.remove_punct(sent)
        sent = self.lemmatize_string(sent)
        sent = self.tokenize_sent(sent)
        sent = self.pad_sent(sent)

        return sent


if __name__ == '__main__':
    pr = PreprocessSents(
        tokenizer_file_path=r"C:\Users\table\PycharmProjects\pajtong\ChatBot\tokenizer.pkl",
        label_encoder_file_path=r"C:\Users\table\PycharmProjects\pajtong\ChatBot\label_encoder.pkl",
        intents_json_file_path="intents.json"
    )
    model = load_model(r"C:\Users\table\PycharmProjects\pajtong\ChatBot\ChatBotIntentsModel.h5")
    while True:
        sent = input()
        sent = pr.full_preprocessing(sent)
        pred = model.predict(sent)
        class_id = np.argmax(pred)
        class_name = pr.decode_class(class_id)
        print(class_name)
        print(pr.get_answer(class_name))