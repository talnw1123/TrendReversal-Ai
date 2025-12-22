import yfinance as yf

def test_ticker(ticker):
    print(f"Testing {ticker}...")
    try:
        dat = yf.Ticker(ticker)
        hist = dat.history(period="1mo")
        if not hist.empty:
            print(f"SUCCESS: {ticker} - {len(hist)} rows")
        else:
            print(f"EMPTY: {ticker}")
    except Exception as e:
        print(f"ERROR: {ticker} - {e}")

tickers_to_test = [
    '^SET.BK', 'SET.BK', '^SET', # Thailand
    'BTC-USD', 'BTC-GBP',        # BTC
    '^GSPC',                     # US (Control)
]

for t in tickers_to_test:
    test_ticker(t)
