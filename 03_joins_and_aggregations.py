# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Joins, Aggregations and Window Functions
# MAGIC
# MAGIC **Focus:** join types and pitfalls · groupBy / agg · window functions
# MAGIC
# MAGIC We enrich trades with the security master and portfolio data, then compute
# MAGIC running positions — the core of any investment data platform.

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, when, lit

RAW_PATH = "/tmp/invest_platform/raw"
read = lambda n: spark.read.option("header", True).option("inferSchema", True).csv(f"{RAW_PATH}/{n}")

trades      = read("trades").withColumn("trade_ts", F.to_timestamp("trade_ts")) \
                            .withColumn("trade_date", F.to_date("trade_ts"))
instruments = read("instruments")
accounts    = read("accounts")
prices      = read("prices").withColumn("price_date", F.to_date("price_date"))
fx          = read("fx_rates").withColumn("rate_date", F.to_date("rate_date"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Join types
# MAGIC
# MAGIC | Type | Keeps |
# MAGIC |---|---|
# MAGIC | `inner` | matches only |
# MAGIC | `left` / `left_outer` | all left rows |
# MAGIC | `right` | all right rows |
# MAGIC | `full` / `outer` | everything |
# MAGIC | `left_semi` | left rows **that have** a match (no right columns) |
# MAGIC | `left_anti` | left rows **with no** match — the data-quality workhorse |
# MAGIC | `cross` | cartesian product — must be explicit |

# COMMAND ----------

enriched = (trades.alias("t")
    .join(instruments.alias("i"), on="instrument_id", how="left")
    .join(accounts.alias("a"),    on="account_id",    how="left")
    .select("t.trade_id", "t.trade_date", "t.account_id", "a.strategy", "a.manager",
            "t.instrument_id", "i.ticker", "i.asset_class", "i.sector", "i.country",
            "t.side", "t.quantity", "t.price", "t.venue", "t.status"))

enriched.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### left_anti — find orphan trades
# MAGIC Trades referencing an instrument that does not exist in the security master. This
# MAGIC is the single most valuable join in data quality work.

# COMMAND ----------

orphans = trades.join(instruments, on="instrument_id", how="left_anti")
print("Orphan trades:", orphans.count())
orphans.select("trade_id", "instrument_id").show(5, False)

# left_semi: trades that DO have a valid instrument, without pulling instrument columns
valid = trades.join(instruments, on="instrument_id", how="left_semi")
print("Valid trades :", valid.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Join pitfalls
# MAGIC
# MAGIC **1. Ambiguous columns.** After a join on differently-named keys, both sides keep
# MAGIC their column. Use `on="col"` (string) to collapse the key, or alias each side.
# MAGIC
# MAGIC **2. Row explosion.** Joining to a non-unique key multiplies rows. Always check
# MAGIC counts before and after.
# MAGIC
# MAGIC **3. Type mismatch.** `"INS001"` never equals `1`. A silent zero-row result is
# MAGIC almost always a type or whitespace mismatch.

# COMMAND ----------

before = trades.count()
after  = enriched.count()
print(f"before={before:,}  after={after:,}  {'OK' if before == after else 'ROW EXPLOSION!'}")

# Multi-key join with different names on each side
px_joined = (trades.alias("t")
    .join(prices.alias("p"),
          (col("t.instrument_id") == col("p.instrument_id")) &
          (col("t.trade_date")    == col("p.price_date")),
          "left")
    .select("t.trade_id", "t.instrument_id", "t.price",
            col("p.close_px").alias("market_close")))

px_joined.show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 groupBy and agg

# COMMAND ----------

by_strategy = (enriched
    .filter(col("status") == "FILLED")
    .withColumn("notional", col("quantity") * col("price"))
    .groupBy("strategy", "asset_class")
    .agg(
        F.count("*").alias("trade_count"),
        F.countDistinct("instrument_id").alias("instruments_traded"),
        F.sum("notional").alias("gross_notional"),
        F.sum(when(col("side") == "BUY", col("notional")).otherwise(-col("notional"))).alias("net_notional"),
        F.avg("price").alias("avg_price"),
        F.expr("percentile_approx(notional, 0.5)").alias("median_notional"),
        F.max("notional").alias("largest_trade"),
        F.min("trade_date").alias("first_trade"),
        F.max("trade_date").alias("last_trade"),
    )
    .orderBy(col("gross_notional").desc()))

by_strategy.show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### pivot — turn rows into columns

# COMMAND ----------

(enriched.filter(col("status") == "FILLED")
    .groupBy("strategy")
    .pivot("side", ["BUY", "SELL"])          # ALWAYS list the values — otherwise Spark scans to find them
    .agg(F.sum(col("quantity") * col("price")).alias("notional"))
    .show(truncate=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Window functions
# MAGIC
# MAGIC A window computes a value **per row** using a set of related rows — without
# MAGIC collapsing the result the way `groupBy` does.
# MAGIC
# MAGIC ```
# MAGIC Window.partitionBy(...)   # the grouping
# MAGIC       .orderBy(...)       # the ordering inside each group
# MAGIC       .rowsBetween(...)   # optional frame
# MAGIC ```

# COMMAND ----------

daily = (enriched
    .filter(col("status") == "FILLED")
    .withColumn("signed_qty", when(col("side") == "BUY", col("quantity")).otherwise(-col("quantity")))
    .groupBy("account_id", "instrument_id", "trade_date")
    .agg(F.sum("signed_qty").alias("net_qty"),
         F.sum(col("quantity") * col("price")).alias("notional")))

w_position = Window.partitionBy("account_id", "instrument_id").orderBy("trade_date")

positions = (daily
    .withColumn("running_position", F.sum("net_qty").over(w_position))
    .withColumn("prev_day_qty",     F.lag("net_qty", 1).over(w_position))
    .withColumn("next_day_qty",     F.lead("net_qty", 1).over(w_position))
    .withColumn("day_change",       col("net_qty") - F.coalesce(col("prev_day_qty"), lit(0)))
    .withColumn("first_trade_date", F.first("trade_date").over(w_position))
    .withColumn("trading_days",     F.count("*").over(w_position)))

positions.orderBy("account_id", "instrument_id", "trade_date").show(12, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ranking — top 5 positions per strategy

# COMMAND ----------

exposure = (enriched
    .filter(col("status") == "FILLED")
    .groupBy("strategy", "instrument_id", "ticker")
    .agg(F.sum(col("quantity") * col("price")).alias("exposure")))

w_rank = Window.partitionBy("strategy").orderBy(col("exposure").desc())

(exposure
    .withColumn("rn",          F.row_number().over(w_rank))     # 1,2,3,4 — no ties
    .withColumn("rnk",         F.rank().over(w_rank))           # 1,2,2,4 — gaps
    .withColumn("dense_rnk",   F.dense_rank().over(w_rank))     # 1,2,2,3 — no gaps
    .withColumn("pct_of_strategy",
                col("exposure") / F.sum("exposure").over(Window.partitionBy("strategy")))
    .filter(col("rn") <= 5)
    .orderBy("strategy", "rn")
    .show(20, False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rolling frames — 5-day moving average price

# COMMAND ----------

w_5day = (Window.partitionBy("instrument_id")
                .orderBy("price_date")
                .rowsBetween(-4, Window.currentRow))

(prices.filter(col("close_px") > 0)
    .withColumn("ma_5d",  F.avg("close_px").over(w_5day))
    .withColumn("vol_5d", F.stddev("close_px").over(w_5day))
    .withColumn("prev_close", F.lag("close_px").over(Window.partitionBy("instrument_id").orderBy("price_date")))
    .withColumn("daily_return", (col("close_px") - col("prev_close")) / col("prev_close"))
    .filter(col("instrument_id") == "INS00001")
    .select("instrument_id", "price_date", "close_px", "ma_5d", "vol_5d", "daily_return")
    .orderBy("price_date").show(15, False))

# COMMAND ----------

# MAGIC %md
# MAGIC **`rowsBetween` vs `rangeBetween`:** `rowsBetween` counts physical rows;
# MAGIC `rangeBetween` uses the *values* of the ordering column. For a true "last 7 calendar
# MAGIC days", cast the date to a Unix timestamp and use `rangeBetween`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Deduplication with a window (the pattern you will reuse everywhere)

# COMMAND ----------

w_dedupe = Window.partitionBy("trade_id").orderBy(col("trade_ts").desc())

deduped = (trades
    .withColumn("rn", F.row_number().over(w_dedupe))
    .filter(col("rn") == 1)
    .drop("rn"))

print(f"raw={trades.count():,}  deduped={deduped.count():,}  removed={trades.count()-deduped.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Which accounts traded an instrument that is **not** in the security master?
# MAGIC    (`left_anti` + `groupBy`)
# MAGIC 2. Convert every trade notional to USD by joining `fx_rates` on currency and date.
# MAGIC    Handle missing rates by carrying the previous available rate forward.
# MAGIC 3. For each trader, rank their trading days by notional and return their busiest day.
# MAGIC 4. Compute a 20-day moving average and flag prices more than 3 standard deviations
# MAGIC    from it.
# MAGIC 5. Compare `row_number`, `rank` and `dense_rank` on ties. When does each matter?
