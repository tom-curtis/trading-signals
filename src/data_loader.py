import ast

import pandas as pd


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.startswith(("b'", 'b"')):
        try:
            text = ast.literal_eval(text).decode("utf-8", errors="ignore")
        except (ValueError, SyntaxError, AttributeError, UnicodeDecodeError):
            text = text[2:-1]

    return " ".join(text.split())


def load_headlines_csv(path):
    df = pd.read_csv(path)

    if not {"Date", "News"}.issubset(df.columns):
        raise ValueError(f"Headlines CSV must contain Date and News columns. Found: {df.columns}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["News"] = df["News"].map(clean_text)

    df = df.dropna(subset=["Date"])
    df = df[df["News"] != ""]

    return df.sort_values("Date").reset_index(drop=True)


def load_prices_csv(path):
    df = pd.read_csv(path)

    required = {"Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Prices CSV missing columns: {required - set(df.columns)}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    return df.sort_values("Date").reset_index(drop=True)


def aggregate_headlines_by_day(headlines_df):
    grouped = (
        headlines_df.groupby("Date", as_index=False)
        .agg(
            text=("News", lambda x: " [SEP] ".join(x)),
            headline_count=("News", "size"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return grouped

def add_price_features(prices_df):
    df = prices_df.copy()

    df["daily_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Open"]
    df["close_vs_prev_close"] = df["Close"].pct_change()
    df["volume_change"] = df["Volume"].pct_change()

    return df


def merge_market_and_headlines(prices_df, headlines_df):
    df = prices_df.merge(
        headlines_df,
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    return df.sort_values("Date").reset_index(drop=True)


def add_next_day_target(df, drop_no_change_days=True):
    df = df.copy()

    df["next_close"] = df["Close"].shift(-1)
    df["next_day_return"] = (df["next_close"] - df["Close"]) / df["Close"]

    df = df.dropna(subset=["next_close", "next_day_return"])

    if drop_no_change_days:
        df = df[df["next_day_return"] != 0]

    df["target"] = (df["next_day_return"] > 0).astype(int)

    return df.reset_index(drop=True)


def split_by_date(df, train_end, val_end):
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    train = df[df["Date"] <= train_end].copy()
    val = df[(df["Date"] > train_end) & (df["Date"] <= val_end)].copy()
    test = df[df["Date"] > val_end].copy()

    return train, val, test