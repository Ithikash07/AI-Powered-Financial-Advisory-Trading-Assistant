#Example Live Trading Code Snippet
'''
def execute_live_trade(symbol, qty, side, order_type="market", time_in_force="gtc"):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,  # 'buy' or 'sell'
            type=order_type,
            time_in_force=time_in_force
        )
        return f"Order submitted successfully: {order}"
    except Exception as e:
        return f"Trade execution failed: {str(e)}"

#2. Continuous Data Ingestion (YouTube & Alpaca News)

import schedule
import time

def update_youtube_advisory_index():
    # 1. Fetch new video transcripts/Q&A
    # 2. Generate embeddings of new content
    # 3. Insert/update those embeddings into vector DB
    print("YouTube advisory content updated")

# Run update every day at midnight
schedule.every().day.at("00:00").do(update_youtube_advisory_index)

# Event loop for scheduler
while True:
    schedule.run_pending()
    time.sleep(60)

'''
# Alpaca News WebSocket for real-time news ingestion


import websocket
import json

def on_message(ws, message):
    news_data = json.loads(message)
    # Extract headline and symbol, run sentiment analysis, update trading signals
    print("Received news:", news_data)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws):
    print("Connection closed")

def on_open(ws):
    subscribe_message = json.dumps({
        "action": "subscribe",
        "symbols": SYMBOLS
    })
    ws.send(subscribe_message)

# Connect to Alpaca's news WebSocket endpoint
ws = websocket.WebSocketApp("wss://stream.data.alpaca.markets/v1beta1/news",
                            on_open=on_open,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

ws.run_forever()
