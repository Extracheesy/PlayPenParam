import pandas as pd
import numpy as np


class TradingSimulator:
    def __init__(self, symbol, timeframe, initial_budget: float, fee: float, direction: str = "long", ma_type="ema"):
        """
        Initialize the TradingSimulator with parameters.

        Parameters
        ----------
        initial_budget : float
            The initial amount of capital to start the simulation with.
        fee : float
            The fraction of the transaction amount to be paid as fee. For example, 0.0001 = 0.01%.
        direction : str
            "long", "short", or "both". Determines the type of trades allowed.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.Trade_Type = direction
        self.ma_type = ma_type
        self.initial_budget = initial_budget
        # self.fee = fee
        self.fee = 0
        self.direction = direction.lower()
        if self.direction not in ["long", "short", "both"]:
            raise ValueError("direction must be 'long', 'short', or 'both'.")

        self.result = None  # Will store results after simulation

    def simulate_trading(self,
                         Close: pd.Series,
                         Entry: pd.Series,
                         Exit: pd.Series,
                         Entry_short: pd.Series,
                         Exit_short: pd.Series) -> dict:
        """
        Simulate the trading strategy using the provided signals.

        Parameters
        ----------
        Close : pd.Series
            Series of closing prices.
        Entry : pd.Series
            Boolean series indicating where to enter a long position.
        Exit : pd.Series
            Boolean series indicating where to exit a long position.
        Entry_short : pd.Series
            Boolean series indicating where to enter a short position.
        Exit_short : pd.Series
            Boolean series indicating where to exit a short position.

        Returns
        -------
        result : dict
            A dictionary containing:
            - "cumulative_return": float
            - "Total Return [%]": float
            - "Benchmark Return [%]": float
            - "Sharpe Ratio": float
            - "End Value": float
            - "Win Rate [%]": float
            - "Total Trades": int
            - "Max Drawdown [%]": float
        """
        capital = self.initial_budget
        position = 0.0  # positive for long, negative for short, 0 if flat
        short_buying_price = 0.0

        portfolio_values = []
        number_of_trades = 0
        number_of_successful_trades = 0

        # Variables to track trades
        in_trade = False
        trade_start_value = 0.0

        for i in range(len(Close)):
            price = Close.iloc[i]

            # Record current portfolio value before action
            portfolio_value = capital + position * price
            portfolio_values.append(portfolio_value)

            # Check exits if currently in a position
            if position > 0:  # Long position
                if self.direction in ["long", "both"] and Exit.iloc[i]:
                    # Exit long
                    proceeds = position * price * (1 - self.fee)
                    capital = proceeds
                    position = 0.0

                    # Complete the trade
                    number_of_trades += 1
                    end_trade_value = capital
                    if end_trade_value > trade_start_value:
                        number_of_successful_trades += 1
                    in_trade = False

            elif position < 0:  # Short position
                if self.direction in ["short", "both"] and Exit_short.iloc[i]:
                    # Exit short (cover) using the provided formula
                    # capital = abs(position)*short_buying_price + (price - short_buying_price)*position
                    capital = abs(position) * short_buying_price + (price - short_buying_price) * position
                    short_buying_price = 0
                    position = 0.0

                    # Complete the trade
                    number_of_trades += 1
                    end_trade_value = capital
                    if end_trade_value > trade_start_value:
                        number_of_successful_trades += 1
                    in_trade = False

            # If flat, check entries
            if position == 0:
                # Long entry
                if (self.direction in ["long", "both"]) and Entry.iloc[i]:
                    if price > 0 and capital > 0:
                        shares = capital / (price * (1 + self.fee))
                        position = shares
                        capital = 0.0
                        # Mark start of trade
                        trade_start_value = portfolio_values[-1]
                        in_trade = True

                # Short entry
                elif (self.direction in ["short", "both"]) and Entry_short.iloc[i]:
                    if price > 0 and capital > 0:
                        shares = capital / (price * (1 + self.fee))
                        position = -shares
                        short_buying_price = price
                        # Proceeds from short sale
                        proceeds = shares * price * (1 - self.fee)
                        capital = proceeds
                        # Mark start of trade
                        trade_start_value = portfolio_values[-1]
                        in_trade = True

            # Update portfolio value after today's actions
            portfolio_values[-1] = capital + position * price

        # Mark to market final value
        final_value = capital + position * Close.iloc[-1]
        cumulative_return = (final_value - self.initial_budget) / self.initial_budget

        # Total Return %
        total_return_percent = cumulative_return * 100.0

        # Benchmark Return %
        benchmark_return = ((Close.iloc[-1] / Close.iloc[0]) - 1) * 100.0

        # Compute Sharpe ratio
        portfolio_values = np.array(portfolio_values)
        if len(portfolio_values) > 1:
            daily_returns = portfolio_values[1:] / portfolio_values[:-1] - 1
        else:
            daily_returns = np.array([])

        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns, ddof=1)
            if std_return == 0:
                sharpe_ratio = np.nan
            else:
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
        else:
            sharpe_ratio = np.nan

        # Compute win rate
        if number_of_trades > 0:
            win_rate = (number_of_successful_trades / number_of_trades) * 100.0
        else:
            win_rate = np.nan

        # Compute Max Drawdown %
        if len(portfolio_values) > 0:
            cumulative_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values / cumulative_max) - 1
            max_drawdown = drawdowns.min() * 100.0  # negative number
            max_drawdown_percent = abs(max_drawdown)
        else:
            max_drawdown_percent = np.nan

        # Store results in self.result
        self.result = {
            'Type': "my_bckt",
            "Symbol": self.symbol,
            "Timeframe": self.timeframe,
            "Trade_Type": self.Trade_Type,
            "MA_Type": self.ma_type,
            "cumulative_return": cumulative_return,
            "Total Return [%]": total_return_percent,
            "Benchmark Return [%]": benchmark_return,
            "Sharpe Ratio": sharpe_ratio,
            "End Value": final_value,
            "Win Rate [%]": win_rate,
            "Total Trades": number_of_trades,
            "Max Drawdown [%]": max_drawdown_percent
        }

        return self.result

    def get_result(self) -> dict:
        """
        Get the simulation results.

        Returns
        -------
        dict
            The dictionary of results if available, otherwise None.
        """
        return self.result

    def get_result_df(self) -> pd.DataFrame:
        """
        Get the simulation results as a pandas DataFrame with a single row.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the results if available, otherwise None.
        """
        if self.result is None:
            return None
        return pd.DataFrame([self.result])
