
import faiss
from sentence_transformers import SentenceTransformer
from alpaca_trade_api import REST
import numpy as np
import yfinance as yf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd

# Alpaca API Setup
API_KEY = "YOUR_ALPACA_API_KEY"
API_SECRET = "YOUR_ALPACA_API_SECRET"
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Moving Average Params
SHORT_WINDOW = 9
LONG_WINDOW = 21

# Sentiment Model Params
VOCAB_SIZE = 5000
MAX_LEN = 50
EMBED_DIM = 100
CLASS_LABELS = ['positive', 'negative', 'neutral']

# ------------------ Advisory Data & Vector DB Setup ------------------
df = pd.read_csv('questions.csv')  

texts = df['answer'].tolist()

advisory_data = [{"id": row.id, "text": row.answer} for row in df.itertuples()]

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings using SentenceTransformer
embeddings = embed_model.encode(texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
id_map = {i: advisory_data[i] for i in range(len(advisory_data))}

def search_advisory(query, top_k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(q_emb, top_k)
    return [id_map[idx]['text'] for idx in indices[0]]

# ------------------ Sentiment Model Setup ------------------

def load_sentiment_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(CLASS_LABELS), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

sentiment_model = load_sentiment_model()

# Assume tokenizer is trained/fitted on your dataset beforehand
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")

def predict_sentiment(headlines):
    sequences = tokenizer.texts_to_sequences(headlines)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    preds = sentiment_model.predict(padded)
    sentiments = [(CLASS_LABELS[np.argmax(p)], float(np.max(p))) for p in preds]
    return sentiments

# ------------------ Alpaca News Fetch ------------------

def fetch_recent_news(symbol, limit=10):
    try:
        news_items = api.get_news(symbol=symbol, limit=limit)
        return [item.headline for item in news_items]
    except Exception as e:
        print(f"Alpaca news fetch error: {e}")
        return []

# ------------------ Technical Indicator ------------------

def get_moving_average_signal(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df['short_ma'] = df['Close'].rolling(SHORT_WINDOW).mean()
    df['long_ma'] = df['Close'].rolling(LONG_WINDOW).mean()
    df['signal'] = 0
    df.loc[(df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1)), 'signal'] = 1
    df.loc[(df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1)), 'signal'] = -1
    return df['signal'].iloc[-1]

# ------------------ Core Integrated Logic ------------------

def handle_user_query_with_advisory(symbol, question):
    # Search advisory vector DB for relevant expert advice
    advice = search_advisory(question)
    # Fetch recent news headlines for symbol
    news = fetch_recent_news(symbol)
    return advice, news

def trade_decision(symbol, question, start_date, end_date):
    # Sentiment analysis on recent news
    news_headlines = fetch_recent_news(symbol)
    if news_headlines:
        sentiments = predict_sentiment(news_headlines)
        pos_count = sum(1 for s, _ in sentiments if s == 'positive')
        neg_count = sum(1 for s, _ in sentiments if s == 'negative')
        sentiment_score = pos_count - neg_count
    else:
        sentiment_score = 0

    # Technical signal from price data
    tech_signal = get_moving_average_signal(symbol, start_date, end_date)

    # Advisory content for user question
    advisory_texts, recent_news = handle_user_query_with_advisory(symbol, question)

    # Combined trade recommendation
    if sentiment_score > 0 and tech_signal == 1:
        recommendation = "BUY"
    elif sentiment_score < 0 and tech_signal == -1:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return {
        "recommendation": recommendation,
        "sentiment_score": sentiment_score,
        "technical_signal": tech_signal,
        "advisory_texts": advisory_texts,
        "recent_news": recent_news
    }

# ------------------ Example Usage ------------------

if __name__ == "__main__":
    user_symbol = "AAPL"
    user_question = "Should I diversify my tech stock investments?"
    backtest_start = "2024-01-01"
    backtest_end = "2024-12-31"

    # Get integrated trade decision and advisory
    result = trade_decision(user_symbol, user_question, backtest_start, backtest_end)

    print(f"Trade Recommendation for {user_symbol}: {result['recommendation']}")
    print(f"Sentiment Score: {result['sentiment_score']}")
    print(f"Technical Signal (MA Crossover): {result['technical_signal']}")

    print("\nAdvisory Texts:")
    for idx, text in enumerate(result['advisory_texts'], 1):
        print(f"{idx}. {text}")

    print("\nRecent News Headlines:")
    for headline in result['recent_news']:
        print("-", headline)
