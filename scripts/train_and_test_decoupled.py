"""
This script no longer uses a model class that abstracts away trading-specific logic including data fetching and feature/labe transformation.

It controls the entire pipeline from data fetching to model training and testing.

During trading time, the trading bot will replicate a similar pipeline.

By decoupling, it makes debugging in downstream systems easier, because downstream now can debug each step independently.

One downside of losing abstraction is potential discrepancies between training/offline research time and online trading time,
because each is implemented in a different system.

"""

import argparse

from utils.bar_fetcher import AlpacaBarFetcher
from utils.datetime_utils import shift_datestr
from utils.script_utils import build_param_overrides, resolve_model_class

def fetch_raw_data(symbol: str, start_datestr: str, end_datestr: str) -> pd.DataFrame:
    bar_fetcher = AlpacaBarFetcher()

    # Extend the fetch date range because due to the nature of time series data, we need further lookback for features for certain point of time.
    # Here we extend minute fetch date range to include the previous day. The assumption is that minute-level features will at most use minute bars since yesterday.
    # For daily data, we extend by 1 year. The assumption is that daily-level features will at most use daily bars since 1 year ago.
    minute_fetch_start_datestr = shift_datestr(start_datestr, -1)
    minute_fetch_end_datestr = end_datestr
    daily_fetch_start_datestr = shift_datestr(start_datestr, -365)
    daily_fetch_end_datestr = shift_datestr(end_datestr, -1)

    minute_bar_df = bar_fetcher.fetch_bars(symbol, minute_fetch_start_datestr, minute_fetch_end_datestr, 'minute')
    daily_bar_df = bar_fetcher.fetch_bars(symbol, daily_fetch_start_datestr, daily_fetch_end_datestr, 'day')

    return minute_bar_df, daily_bar_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--train_start', type=str, required=False, default='2026-01-01')
    parser.add_argument('--train_end', type=str, required=False, default='2026-03-31')
    parser.add_argument('--test_start', type=str, required=False, default='2026-04-01')
    parser.add_argument('--test_end', type=str, required=False, default='2026-04-30')
    parser.add_argument('--param', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()

    # Get custom parameters to initialize model
    init_params = build_param_overrides(args.param)

    # Fetch raw data
    minute_bar_df, daily_bar_df = fetch_raw_data(args.symbol, args.train_start, args.train_end)

    # Convert raw data to ML-ready features and labels
    pass

    model_class = resolve_model_class(args.model)
    model = model_class(**init_params)

    training_metrics, validation_metrics = model.train(args.train_start, args.train_end)
    testing_metrics = model.test(args.test_start, args.test_end)

    print(f"Training metrics: {training_metrics}")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Testing metrics: {testing_metrics}")
