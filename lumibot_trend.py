from config import ALPACA_CONFIG
from datetime import datetime, timedelta

from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.strategies import Strategy
from lumibot.traders import Trader

import numpy as np
import pandas as pd

class Trend(Strategy):

    def initialize(self):
        signal = None
        start = "2023-04-15"

        self.signal = signal
        self.start = start
        self.sleeptime = "1D"

#Simple moving average strategy
    def on_trading_iteration(self):
        bars = self.get_historical_prices("AAPL", 22, "day")
        gld = bars.df
        #gld = pd.DataFrame(yf.download("GLD", self.start)['Close'])
        gld['9-day'] = gld['close'].rolling(9).mean()
        gld['21-day'] = gld['close'].rolling(21).mean()
        gld['Signal'] = np.where(np.logical_and(gld['9-day'] > gld['21-day'],
                                                gld['9-day'].shift(1) < gld['21-day'].shift(1)),
                                 "BUY", None)
        gld['Signal'] = np.where(np.logical_and(gld['9-day'] < gld['21-day'],
                                                gld['9-day'].shift(1) > gld['21-day'].shift(1)),
                                 "SELL", gld['Signal'])
        self.signal = gld.iloc[-1].Signal
        
        symbol = "AAPL"
        quantity = 200
        if self.signal == 'BUY':
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()
                
            order = self.create_order(symbol, quantity, "buy")
            self.submit_order(order)

        elif self.signal == 'SELL':
            pos = self.get_position(symbol)
            if pos is not None:
                self.sell_all()
                
            order = self.create_order(symbol, quantity, "sell")
            self.submit_order(order)

if __name__ == "__main__":
    #live trading is complex since it requires one or two days to check whether the order is placed
    trade = False
    if trade:
        broker = Alpaca(ALPACA_CONFIG)
        strategy = Trend(broker=broker)
        bot = Trader()
        bot.add_strategy(strategy)
        bot.run_all()
    else:
        start = datetime(2023, 4, 15)
        end = datetime(2024, 4, 4)
        Trend.backtest(
            YahooDataBacktesting,
            start,
            end,
            parameters={'symbol':'AAPL','cash_at_risk':.50}
        )