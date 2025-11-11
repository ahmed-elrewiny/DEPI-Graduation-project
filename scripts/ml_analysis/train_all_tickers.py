import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns and not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Daily_Return" not in df.columns or df["Daily_Return"].isna().all():
        if "Close" in df.columns:
            df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()
        else:
            df["Daily_Return"] = np.nan
    if "Volatility_30" not in df.columns or df["Volatility_30"].isna().all():
        df["Volatility_30"] = df.groupby("Ticker")["Daily_Return"].transform(lambda s: s.rolling(30).std())
    # Basic moving averages
    if "MA_7" not in df.columns or df["MA_7"].isna().all():
        df["MA_7"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(7).mean())
    if "MA_30" not in df.columns or df["MA_30"].isna().all():
        df["MA_30"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(30).mean())
    if "MA_50" not in df.columns or df["MA_50"].isna().all():
        df["MA_50"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(50).mean())
    # EMA and MACD
    if "EMA_12" not in df.columns or df["EMA_12"].isna().all():
        df["EMA_12"] = df.groupby("Ticker")["Close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    if "EMA_26" not in df.columns or df["EMA_26"].isna().all():
        df["EMA_26"] = df.groupby("Ticker")["Close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    if "MACD" not in df.columns or df["MACD"].isna().all():
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
    if "Signal_Line" not in df.columns or df["Signal_Line"].isna().all():
        df["Signal_Line"] = df.groupby("Ticker")["MACD"].transform(lambda s: s.ewm(span=9, adjust=False).mean())

    # Bands
    if "Rolling_Std_20" not in df.columns or df["Rolling_Std_20"].isna().all():
        df["Rolling_Std_20"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(20).std())
    if "Upper_Band" not in df.columns or df["Upper_Band"].isna().all():
        df["Upper_Band"] = df["MA_7"] + 2 * df["Rolling_Std_20"]
    if "Lower_Band" not in df.columns or df["Lower_Band"].isna().all():
        df["Lower_Band"] = df["MA_7"] - 2 * df["Rolling_Std_20"]

    # Targets
    df["Next_Close"] = df.groupby("Ticker")["Close"].shift(-1)
    df["Direction"] = (df["Next_Close"] > df["Close"]).astype(int)
    if "Monthly_Return" not in df.columns or df["Monthly_Return"].isna().all():
        df["Monthly_Return"] = df.groupby("Ticker")["Close"].transform(lambda s: s.pct_change(30))
    if "Monthly_Volatility" not in df.columns or df["Monthly_Volatility"].isna().all():
        df["Monthly_Volatility"] = df.groupby("Ticker")["Daily_Return"].transform(lambda s: s.rolling(30).std())

    return df


def train_and_predict_for_ticker(ticker_df: pd.DataFrame) -> pd.DataFrame:
    ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
    # Features inspired by the analysis script, include only columns present
    candidate_features = [
        "Close", "Daily_Return", "MA_7", "MA_30", "Volatility_30",
        "MA_50", "EMA_12", "EMA_26", "MACD", "Signal_Line",
        "Rolling_Std_20", "Upper_Band", "Lower_Band"
    ]
    features = [c for c in candidate_features if c in ticker_df.columns]
    needed_targets = ["Next_Close", "Direction", "Monthly_Return", "Monthly_Volatility"]
    if any(c not in ticker_df.columns for c in features + needed_targets):
        return pd.DataFrame(columns=[
            "Date","Ticker","Actual_Next_Close","GBR_Pred_Next_Close","RFC_Pred_Next_Close",
            "Actual_Direction","RFC_Pred_Direction","GBC_Pred_Direction",
            "Actual_Monthly_Return","Pred_Monthly_Return",
            "Actual_Monthly_Volatility","Pred_Monthly_Volatility"
        ])

    model_df = ticker_df.dropna(subset=features + needed_targets).copy()
    if len(model_df) < 200:
        # not enough data to train — skip
        return pd.DataFrame(columns=[
            "Date","Ticker","Actual_Next_Close","GBR_Pred_Next_Close","RFC_Pred_Next_Close",
            "Actual_Direction","RFC_Pred_Direction","GBC_Pred_Direction",
            "Actual_Monthly_Return","Pred_Monthly_Return",
            "Actual_Monthly_Volatility","Pred_Monthly_Volatility"
        ])

    X = model_df[features].values
    y_next_close = model_df["Next_Close"].values
    y_direction = model_df["Direction"].values
    y_monthly_return = model_df["Monthly_Return"].values
    y_monthly_volatility = model_df["Monthly_Volatility"].values

    # chronological split 70/30
    split_idx = int(0.7 * len(model_df))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_nc_train, y_nc_test = y_next_close[:split_idx], y_next_close[split_idx:]
    y_dir_train, y_dir_test = y_direction[:split_idx], y_direction[split_idx:]
    y_mr_train, y_mr_test = y_monthly_return[:split_idx], y_monthly_return[split_idx:]
    y_mv_train, y_mv_test = y_monthly_volatility[:split_idx], y_monthly_volatility[split_idx:]

    # Models
    gbr_next_close = GradientBoostingRegressor(random_state=42)
    rfr_next_close = RandomForestRegressor(n_estimators=200, random_state=42)
    rfc_direction = RandomForestClassifier(n_estimators=300, random_state=42)
    gbc_direction = GradientBoostingClassifier(random_state=42)
    gbr_monthly_return = GradientBoostingRegressor(random_state=42)
    gbr_monthly_volatility = GradientBoostingRegressor(random_state=42)

    # Train
    gbr_next_close.fit(X_train, y_nc_train)
    rfr_next_close.fit(X_train, y_nc_train)
    rfc_direction.fit(X_train, y_dir_train)
    gbc_direction.fit(X_train, y_dir_train)
    gbr_monthly_return.fit(X_train, y_mr_train)
    gbr_monthly_volatility.fit(X_train, y_mv_train)

    # Predict for test segment
    preds_nc_gbr = gbr_next_close.predict(X_test)
    preds_nc_rfr = rfr_next_close.predict(X_test)
    preds_dir_rfc = rfc_direction.predict(X_test)
    preds_dir_gbc = gbc_direction.predict(X_test)
    preds_mr = gbr_monthly_return.predict(X_test)
    preds_mv = gbr_monthly_volatility.predict(X_test)

    out = pd.DataFrame({
        "Date": model_df.iloc[split_idx:]["Date"].values,
        "Ticker": model_df.iloc[split_idx:]["Ticker"].values,
        "Actual_Next_Close": y_nc_test,
        "GBR_Pred_Next_Close": preds_nc_gbr,
        "RFC_Pred_Next_Close": preds_nc_rfr,
        "Actual_Direction": y_dir_test,
        "RFC_Pred_Direction": preds_dir_rfc,
        "GBC_Pred_Direction": preds_dir_gbc,
        "Actual_Monthly_Return": y_mr_test,
        "Pred_Monthly_Return": preds_mr,
        "Actual_Monthly_Volatility": y_mv_test,
        "Pred_Monthly_Volatility": preds_mv,
    })
    return out


def main():
    # Paths relative to this script directory
    here = os.path.dirname(__file__)
    clean_path = os.path.join(here, "..", "data", "clean", "clean_tech_stocks.csv")
    output_dir = os.path.join(here, "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "predictions.csv")

    print(f"Loading data from: {clean_path}")
    df = pd.read_csv(clean_path)
    df = ensure_features(df)

    all_preds = []
    tickers = sorted(df["Ticker"].dropna().unique())
    print(f"Training across {len(tickers)} tickers...")
    for t in tickers:
        tdf = df[df["Ticker"] == t]
        preds_t = train_and_predict_for_ticker(tdf)
        if not preds_t.empty:
            all_preds.append(preds_t)
            print(f"[{t}] rows: {len(preds_t)}")
        else:
            print(f"[{t}] skipped (insufficient data)")

    if not all_preds:
        print("No predictions were generated. Please check input data.")
        return

    result = pd.concat(all_preds, ignore_index=True).sort_values(["Ticker", "Date"])
    result.to_csv(output_path, index=False)
    print(f"\n✅ Saved predictions to: {output_path}")
    print(f"Rows: {len(result)} | Tickers covered: {result['Ticker'].nunique()}")


if __name__ == "__main__":
    main()


