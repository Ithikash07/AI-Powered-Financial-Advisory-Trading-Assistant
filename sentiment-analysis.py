import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from alpaca_trade_api import REST

API_KEY = "your_alpaca_api_key"
API_SECRET = "your_alpaca_api_secret"
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def fetch_alpaca_news(symbol: str, start: str, end: str, limit:int=100):
    news = api.get_news(symbol=symbol, start=start, end=end, limit=limit)
    return [item.headline for item in news]

# List of companies' stock symbols
symbols = ["AAPL", "TSLA", "MSFT"]

all_news = []
for symbol in symbols:
    news = fetch_alpaca_news(symbol, "2023-01-01", "2023-01-31")
    all_news.extend(news)

# For demonstration, assign neutral label (2) to all news temporarily
labels = [2] * len(all_news)

# Tokenization and padding
vocab_size = 5000
max_len = 50
embedding_dim = 100
num_classes = 3

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(all_news)
sequences = tokenizer.texts_to_sequences(all_news)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

labels_categorical = tf.keras.utils.to_categorical(labels, num_classes)

# Build and compile LSTM sentiment classification model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model (with dummy labels here)
model.fit(padded_sequences, labels_categorical, epochs=10, batch_size=8)

# Predict function example
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)
    classes = ['positive', 'negative', 'neutral']
    return classes[np.argmax(pred)], float(np.max(pred))

test_news = "Tesla reports record profits amid rising demand"
sentiment, confidence = predict_sentiment(test_news)
print(f"Predicted sentiment: {sentiment} with confidence {confidence:.4f}")
