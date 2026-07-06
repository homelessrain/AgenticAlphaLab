from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd

from utils.bar_fetcher import AlpacaBarFetcher
from utils.feature_transformer import OLHCVFeatureTransformer
from utils.label_transformer import DirectionLabelTransformer


class MLDataProcessor(ABC):
    """
    Base class for processing raw bar data into ML-ready features and labels.
    """
    def __init__(self):
        pass

    @abstractmethod
    def run(self, start_datestr: str, end_datestr: str) -> pd.DataFrame:
        """
        Fetch and process bar data into ML-ready features and labels for the
        given date range.
        """
        pass


class MinuteAndDailyMLDataProcessor(MLDataProcessor):
    """
    Processes minute and daily bar data into ML-ready features and labels.

    Minute bars provide both labels (via lookforward) and short-term features.
    Daily bars provide longer-horizon features joined on the previous trading day.
    """
    def __init__(
        self,
        symbol: str,
        bar_fetcher=None,
        label_transformer=None,
        minute_feature_transformer=None,
        daily_feature_transformer=None,
        generate_labels=True,
    ):
        super().__init__()

        self._symbol = symbol
        self._bar_fetcher = bar_fetcher if bar_fetcher is not None else AlpacaBarFetcher()
        self._minute_label_transformer = (
            label_transformer if label_transformer is not None else DirectionLabelTransformer()
        ) if generate_labels else None
        self._minute_feature_transformer = minute_feature_transformer or OLHCVFeatureTransformer()
        self._daily_feature_transformer  = daily_feature_transformer  or OLHCVFeatureTransformer()

    @property
    def label_column(self) -> str:
        return self._minute_label_transformer.label_column if self._minute_label_transformer is not None else None

    @property
    def feature_columns(self) -> list[str]:
        # The minute/daily merge uses suffixes=('_minute', '_daily'), so mirror that here.
        return (
            [f'{c}_minute' for c in self._minute_feature_transformer.feature_columns] +
            [f'{c}_daily'  for c in self._daily_feature_transformer.feature_columns]
        )

    @staticmethod
    def _flatten(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Drop the symbol level of a MultiIndex bar df and expose timestamp as 'datetime'."""
        return (
            df.xs(symbol, level='symbol')
              .reset_index()
              .rename(columns={'timestamp': 'datetime'})
        )

    def run(self, start_datestr: str, end_datestr: str, drop_label_na: bool = True) -> pd.DataFrame:
        """
        Process minute and daily bar data into ML-ready features and labels.
        """
        # Extend the fetch range so rolling windows and lookforward labels
        # are fully populated within (start_datestr, end_datestr).
        minute_fetch_start = (datetime.strptime(start_datestr, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        minute_fetch_end   = (datetime.strptime(end_datestr,   '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        # Daily bars need a longer lookback for slow MAs (e.g. MA-200).
        daily_fetch_start  = (datetime.strptime(start_datestr, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')

        minute_bars_df = self._bar_fetcher.fetch_bars(
            self._symbol, minute_fetch_start, minute_fetch_end, timeframe='minute',
        )
        daily_bars_df = self._bar_fetcher.fetch_bars(
            self._symbol, daily_fetch_start, end_datestr, timeframe='day',
        )

        minute_df = self._flatten(minute_bars_df, self._symbol)
        daily_df  = self._flatten(daily_bars_df,  self._symbol)

        print(f"Minute bars df shape: {minute_df.shape}")
        print(f"Daily bars df shape: {daily_df.shape}")

        # Labels (minute-level)
        if self._minute_label_transformer is not None:
            minute_label_df = self._minute_label_transformer.transform(minute_df.copy())
        else:
            minute_label_df = None

        # Features (minute-level)
        minute_feature_df = self._minute_feature_transformer.transform(minute_df.copy())
        print(f"Minute feature df shape after minute feature transformer: {minute_feature_df.shape}")

        # Features (daily-level)
        daily_feature_df = self._daily_feature_transformer.transform(daily_df.copy())
        print(f"Daily feature df shape after daily feature transformer: {daily_feature_df.shape}")

        # Pre-rename feature columns with their intended suffixes BEFORE merging.
        # Pandas merge only adds suffixes to columns that collide between both sides.
        # When minute and daily transformers differ (e.g. DailyOLHCVFeatureTransformer
        # omits time features), columns unique to one side would be left un-suffixed,
        # causing a mismatch with the '_minute'/'_daily' names returned by feature_columns.
        minute_feature_df = minute_feature_df.rename(
            columns={c: f'{c}_minute' for c in self._minute_feature_transformer.feature_columns}
        )
        daily_feature_df = daily_feature_df.rename(
            columns={c: f'{c}_daily' for c in self._daily_feature_transformer.feature_columns}
        )

        # Join daily features onto the *next* trading day so we never leak
        # same-day daily information into minute-level rows.
        daily_feature_df['date']              = daily_feature_df['datetime'].dt.date
        daily_feature_df['next_trading_date'] = daily_feature_df['date'].shift(-1)
        daily_feature_df = daily_feature_df.drop(columns=['datetime'])

        minute_feature_df['date'] = minute_feature_df['datetime'].dt.date

        merged_feature_df = pd.merge(
            minute_feature_df, daily_feature_df,
            left_on='date', right_on='next_trading_date',
            how='left',
            suffixes=('_minute', '_daily'),  # fallback suffix for any remaining collisions (e.g. raw OHLCV cols)
        )
        # Restore 'datetime' as a plain column for downstream merges and filtering.
        merged_feature_df = merged_feature_df.rename(columns={'datetime_minute': 'datetime'})

        print(f"Merged feature df shape after merging minute and daily features: {merged_feature_df.shape}")

        # Combine labels and features
        if minute_label_df is not None:
            merged_df = pd.merge(minute_label_df, merged_feature_df, on='datetime', how='left')
        else:
            merged_df = merged_feature_df

        print(f"Merged df shape after merging features and labels: {merged_df.shape}")

        # Filter to the requested date range
        merged_df['datestr'] = merged_df['datetime'].dt.strftime('%Y-%m-%d')
        print(f'min datestr in merged_df: {merged_df["datestr"].min()}, max datestr in merged_df: {merged_df["datestr"].max()}')
        print(f'start_datestr: {start_datestr}, end_datestr: {end_datestr}')
        filtered_df = merged_df[
            (merged_df['datestr'] >= start_datestr) &
            (merged_df['datestr'] <= end_datestr)
        ]

        print(f"Filtered df shape after filtering to requested date range: {filtered_df.shape}")

        if drop_label_na and self._minute_label_transformer is not None:
            filtered_df = filtered_df.dropna(subset=[self.label_column])

        print(f"Filtered df shape after dropping rows with NA labels: {filtered_df.shape}")

        return filtered_df


class MultipleSymbolMinuteAndDailyMLDataProcessor(MLDataProcessor):
    """
    Runs MinuteAndDailyMLDataProcessor for multiple symbols and joins the results on datetime.
    """
    def __init__(
        self,
        symbols: list[str],
        list_of_ml_data_processors: list[MinuteAndDailyMLDataProcessor],
    ):
        super().__init__()

        assert len(symbols) == len(list_of_ml_data_processors), \
            "Number of symbols and number of ML data processors must be the same"

        self._symbols = symbols
        self._list_of_ml_data_processors = list_of_ml_data_processors
    
    @property
    def label_column(self) -> str:
        # Get the first non-None label column
        for processor in self._list_of_ml_data_processors:
            if processor.label_column is not None:
                return processor.label_column
        return None

    @property
    def feature_columns(self) -> list[str]:
        # The first symbol's columns are unsuffixed; subsequent symbols get _{symbol} appended
        # by the merge in run(), so we must mirror that here.
        result = []
        for i, (symbol, processor) in enumerate(zip(self._symbols, self._list_of_ml_data_processors)):
            if i == 0:
                result.extend(processor.feature_columns)
            else:
                result.extend(f'{col}_{symbol}' for col in processor.feature_columns)
        return result

    def run(self, start_datestr: str, end_datestr: str, drop_label_na: bool = True) -> pd.DataFrame:
        """
        Run the ML data processor for multiple symbols and join the results on datetime.

        Return a dataframe with both raw columns and ML-ready features and labels. Downstream code should use the label_column and feature_columns properties to access the columns.
        """
        merged_df = None
        for symbol, processor in zip(self._symbols, self._list_of_ml_data_processors):
            symbol_df = processor.run(start_datestr, end_datestr, drop_label_na=drop_label_na)
            print(f"Symbol: {symbol}, Shape: {symbol_df.shape}")
            if merged_df is None:
                merged_df = symbol_df
            else:
                merged_df = pd.merge(merged_df, symbol_df, on='datetime', how='left', suffixes=('', f'_{symbol}'))
            print(f"Merged shape: {merged_df.shape}")

        return merged_df
