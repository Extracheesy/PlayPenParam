import ccxt
import pandas as pd
import datetime
import backtrader as bt


def fetch_binance_data(symbol, timeframe, start_date, end_date):
    binance = ccxt.binance()
    since = binance.parse8601(f'{start_date}T00:00:00Z')
    end = binance.parse8601(f'{end_date}T00:00:00Z')
    all_ohlcv = []
    while since < end:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        since = ohlcv[-1][0] + 1
        all_ohlcv.extend(ohlcv)
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


# Paramètres
symbol = 'BTC/USDT'
timeframe = '5m'
start_date = '2024-03-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')

# Récupération des données
data = fetch_binance_data(symbol, timeframe, start_date, end_date)

# Vérifiez les données
print(data.head())  # Debugging: Vérifiez que les données sont correctement chargées

# Initialisation de Backtrader
cerebro = bt.Cerebro()


class GridTradingStrategy(bt.Strategy):
    params = (
        ('grid_size', 1000),
        ('grid_high', 80000),
        ('grid_low', 50000),
        ('size_per_trade', 0.001),
    )

    def __init__(self):
        self.grid_levels = list(range(self.params.grid_low, self.params.grid_high, self.params.grid_size))
        self.open_orders = {}

    def next(self):
        current_price = self.data.close[0]

        for level in self.grid_levels:
            if level not in self.open_orders:
                if current_price <= level:
                    self.open_orders[level] = self.buy(size=self.params.size_per_trade, exectype=bt.Order.Limit,
                                                       price=level)
                elif current_price >= level:
                    self.open_orders[level] = self.sell(size=self.params.size_per_trade, exectype=bt.Order.Limit,
                                                        price=level)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            level = order.info.get('level', None)
            if level in self.open_orders:
                del self.open_orders[level]

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f'Profit: {trade.pnl:.2f} USD')


cerebro.addstrategy(GridTradingStrategy)

# Conversion des données pour Backtrader
data_bt = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data_bt)

cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)

print(f'Valeur du portefeuille au départ: {cerebro.broker.getvalue():.2f} USD')
cerebro.run()
print(f'Valeur du portefeuille à la fin: {cerebro.broker.getvalue():.2f} USD')

cerebro.plot()
