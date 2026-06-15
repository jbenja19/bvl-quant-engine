import yfinance as yf
import pandas as pd
import numpy as np

def get_bvl_data():
    """
    Downloads daily Close price and Volume for the 15 most representative BVL tickers
    for the last 5 years. Cleans the data by forward-filling prices and setting 
    volume to 0 on non-trading days.
    """
    tickers = [
        "BAP.LM", "SCCO.LM", "BVN.LM", "IFS.LM", "ALICORC1.LM",
        "CPACASC1.LM", "FERREYC1.LM", "VOLCABC1.LM", "BACKUSI1.LM", "BBVAC1.LM",
        "CORAREI1.LM", "UNACEMC1.LM", "LUSURC1.LM", "INRETC1.LM", "CREDITC1.LM"
    ]
    
    print("Downloading 5 years of daily data from Yahoo Finance...")
    # Download using group_by='ticker' to easily separate series
    df_raw = yf.download(tickers, period="5y", group_by="ticker", progress=False)
    
    close_list = []
    volume_list = []
    
    for ticker in tickers:
        # Check if the ticker exists in the downloaded data
        if ticker in df_raw.columns.levels[0]:
            ticker_df = df_raw[ticker]
            # Copy series to avoid modifications to slice views
            close_series = ticker_df["Close"].copy()
            volume_series = ticker_df["Volume"].copy()
        else:
            # Fallback if ticker didn't download correctly
            print(f"Warning: Ticker {ticker} not found in Yahoo Finance results. Creating empty series.")
            close_series = pd.Series(np.nan, index=df_raw.index)
            volume_series = pd.Series(np.nan, index=df_raw.index)
        
        # Identify dates where there was no negotiation (price is NaN)
        no_trade_mask = close_series.isna()
        
        # Do NOT forward fill prices; keep them as NaN so they are skipped properly in log returns
        # close_series = close_series.ffill().bfill()
        
        # 2. Impute volumes: set to 0 where price was NaN or where volume is NaN
        volume_series = volume_series.fillna(0.0)
        volume_series.loc[no_trade_mask] = 0.0
        
        close_list.append(close_series.rename(ticker))
        volume_list.append(volume_series.rename(ticker))
        
    df_close = pd.concat(close_list, axis=1)
    df_volume = pd.concat(volume_list, axis=1)
    
    # Structure columns as MultiIndex: (Metric, Ticker)
    df_close.columns = pd.MultiIndex.from_product([["Close"], df_close.columns])
    df_volume.columns = pd.MultiIndex.from_product([["Volume"], df_volume.columns])
    
    consolidated = pd.concat([df_close, df_volume], axis=1)
    return consolidated

if __name__ == "__main__":
    df = get_bvl_data()
    print("\n--- Consolidated Data shape ---")
    print(df.shape)
    print("\n--- Volume Descriptive Statistics ---")
    print(df["Volume"].describe())
    print("\n--- Missing Price Check (Should be 0) ---")
    print(df["Close"].isna().sum())
