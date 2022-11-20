import pandas as pd
from ChatBot.chatbotPipelines.Pipelines import setup_prediction_pipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

to_pred_df = pd.DataFrame({
    "X": ["Hi", "How are you?", "cut", "How do you treat a Headache?", "My name is Sarah", "How to treat a fever?"]
})


prediction_pipeline = setup_prediction_pipeline(
    target_column_name="X",
    max_length=10,
    tokenizer_path=r"data/tokenizer.pkl",
    label_encoder_path=r"data/label_encoder.pkl",
    unique_words_path=f"data/words.pkl",
    model_path="Test.h5",
    intents_path="../intents4.json"
)

data = prediction_pipeline.fit_transform(to_pred_df)
print(data)