# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup: Generate the Investment Dataset
# MAGIC
# MAGIC **Run this notebook once before anything else.** Every other notebook in this
# MAGIC package reads the raw files it produces.
# MAGIC
# MAGIC ### The business scenario
# MAGIC We are building the data platform for a mid-size investment manager. Four source
# MAGIC systems feed us:
# MAGIC
# MAGIC | Source | Grain | Arrives as |
# MAGIC |---|---|---|
# MAGIC | Order Management System | one row per trade execution | CSV, hourly drops |
# MAGIC | Security Master | one row per instrument | CSV, daily snapshot |
# MAGIC | Market Data vendor | one row per instrument per day | CSV, end of day |
# MAGIC | Portfolio Accounting | one row per account | CSV, daily snapshot |
# MAGIC
# MAGIC Deliberate defects are baked into the generated data — nulls, duplicates,
# MAGIC negative prices, unknown instrument ids, late-arriving trades. You will fix
# MAGIC them in notebooks 06 and 10.

# COMMAND ----------

# MAGIC %md ## Parameters

# COMMAND ----------

BASE_PATH   = "/tmp/invest_platform"        # change to a Volume or ADLS/S3 path if you prefer
RAW_PATH    = f"{BASE_PATH}/raw"
DELTA_PATH  = f"{BASE_PATH}/delta"
CATALOG     = "invest_demo"                  # used from notebook 05 onwards

N_TRADES        = 200_000
N_INSTRUMENTS   = 500
N_ACCOUNTS      = 40
PRICE_DAYS      = 120

print(f"Raw path   : {RAW_PATH}")
print(f"Delta path : {DELTA_PATH}")

# COMMAND ----------

# MAGIC %md ## Generate

# COMMAND ----------

import random, datetime as dt
from pyspark.sql import functions as F, Row

random.seed(42)

SECTORS      = ["Technology", "Financials", "Energy", "Healthcare", "Industrials", "Consumer"]
ASSET_CLASS  = ["EQUITY", "BOND", "ETF", "FUTURE"]
COUNTRIES    = ["US", "GB", "DE", "JP", "IN"]
VENUES       = ["NYSE", "NASDAQ", "LSE", "XETRA", "TSE", "NSE"]
STRATEGIES   = ["Long/Short Equity", "Macro", "Event Driven", "Quant Systematic"]

# ---- instruments (security master) -------------------------------------------------
instruments = []
for i in range(1, N_INSTRUMENTS + 1):
    instruments.append(Row(
        instrument_id = f"INS{i:05d}",
        ticker        = f"TCK{i:04d}",
        instrument_name = f"Instrument {i} Holdings PLC",
        asset_class   = random.choice(ASSET_CLASS),
        sector        = random.choice(SECTORS) if random.random() > 0.02 else None,   # 2% nulls
        country       = random.choice(COUNTRIES),
        currency      = random.choice(["USD", "USD", "USD", "GBP", "EUR", "JPY", "INR"]),
        lot_size      = random.choice([1, 10, 100]),
    ))
df_instruments = spark.createDataFrame(instruments)

# ---- accounts / portfolios ---------------------------------------------------------
accounts = []
for a in range(1, N_ACCOUNTS + 1):
    accounts.append(Row(
        account_id    = f"ACC{a:04d}",
        portfolio_name= f"Portfolio {a}",
        strategy      = random.choice(STRATEGIES),
        manager       = f"PM{random.randint(1, 8):02d}",
        base_currency = "USD",
        aum_usd       = round(random.uniform(5e6, 8e8), 2),
        opened_date   = str(dt.date(2019, 1, 1) + dt.timedelta(days=random.randint(0, 1800))),
    ))
df_accounts = spark.createDataFrame(accounts)

# ---- prices (market data) ----------------------------------------------------------
start_price_date = dt.date.today() - dt.timedelta(days=PRICE_DAYS)
prices = []
for ins in instruments:
    px = random.uniform(20, 400)
    for d in range(PRICE_DAYS):
        pdate = start_price_date + dt.timedelta(days=d)
        if pdate.weekday() >= 5:                      # skip weekends
            continue
        px = max(1.0, px * (1 + random.gauss(0, 0.018)))
        close = round(px, 4)
        prices.append(Row(
            instrument_id = ins.instrument_id,
            price_date    = str(pdate),
            open_px       = round(close * random.uniform(0.98, 1.02), 4),
            high_px       = round(close * random.uniform(1.00, 1.04), 4),
            low_px        = round(close * random.uniform(0.96, 1.00), 4),
            close_px      = close if random.random() > 0.003 else -close,   # 0.3% bad prices
            volume        = random.randint(1_000, 5_000_000),
        ))
df_prices = spark.createDataFrame(prices)

# ---- trades (order management system) ----------------------------------------------
start_trade_ts = dt.datetime.now() - dt.timedelta(days=30)
trades = []
for t in range(1, N_TRADES + 1):
    ins   = random.choice(instruments)
    acct  = random.choice(accounts)
    ts    = start_trade_ts + dt.timedelta(seconds=random.randint(0, 30 * 24 * 3600))
    qty   = random.choice([100, 250, 500, 1000, 2500, 5000, 10000])
    trades.append(Row(
        trade_id      = f"TRD{t:08d}",
        trade_ts      = ts.strftime("%Y-%m-%d %H:%M:%S"),
        account_id    = acct.account_id,
        instrument_id = ins.instrument_id if random.random() > 0.005 else "INS99999",  # orphans
        side          = random.choice(["BUY", "SELL"]),
        quantity      = qty if random.random() > 0.004 else None,                      # nulls
        price         = round(random.uniform(20, 400), 4),
        currency      = ins.currency,
        trader_id     = f"TDR{random.randint(1, 25):03d}",
        venue         = random.choice(VENUES),
        status        = random.choices(["FILLED", "PARTIAL", "CANCELLED"], [0.9, 0.07, 0.03])[0],
        source_system = "OMS",
    ))

# inject ~1% exact duplicates — a real replay artefact from the OMS
trades += random.sample(trades, k=int(N_TRADES * 0.01))
df_trades = spark.createDataFrame(trades)

# ---- fx rates ----------------------------------------------------------------------
fx = []
for d in range(PRICE_DAYS):
    fdate = start_price_date + dt.timedelta(days=d)
    for ccy, base in [("USD", 1.0), ("GBP", 1.27), ("EUR", 1.08), ("JPY", 0.0067), ("INR", 0.012)]:
        fx.append(Row(rate_date=str(fdate), currency=ccy,
                      usd_rate=round(base * random.uniform(0.99, 1.01), 6)))
df_fx = spark.createDataFrame(fx)

# COMMAND ----------

# MAGIC %md ## Write raw CSV files

# COMMAND ----------

def write_csv(df, name, partitions=1):
    (df.repartition(partitions)
       .write.mode("overwrite")
       .option("header", True)
       .csv(f"{RAW_PATH}/{name}"))
    print(f"{name:<14} rows={df.count():>8,}")

write_csv(df_instruments, "instruments")
write_csv(df_accounts,    "accounts")
write_csv(df_prices,      "prices", partitions=4)
write_csv(df_trades,      "trades", partitions=8)
write_csv(df_fx,          "fx_rates")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Known data defects (you will handle these later)
# MAGIC
# MAGIC | Defect | Table | Rate | Fixed in |
# MAGIC |---|---|---|---|
# MAGIC | Exact duplicate trades | trades | ~1% | 06, 12 |
# MAGIC | Null quantity | trades | ~0.4% | 06, 10 |
# MAGIC | Orphan instrument_id (`INS99999`) | trades | ~0.5% | 10 |
# MAGIC | Negative close price | prices | ~0.3% | 10 |
# MAGIC | Null sector | instruments | ~2% | 06 |
# MAGIC
# MAGIC Setup complete — continue to `01_pyspark_fundamentals`.
