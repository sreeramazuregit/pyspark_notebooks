# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Transformations
# MAGIC
# MAGIC **Focus:** `withColumn` · `when/otherwise` · `cast` · string and date functions
# MAGIC
# MAGIC We turn raw OMS trade records into an analyst-ready shape: typed, enriched,
# MAGIC standardised.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import *

BASE_PATH = "/tmp/invest_platform"
RAW_PATH  = f"{BASE_PATH}/raw"

trades = (spark.read.option("header", True).option("inferSchema", True)
          .csv(f"{RAW_PATH}/trades"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 withColumn — add and replace
# MAGIC
# MAGIC `withColumn(name, expr)` returns a **new** DataFrame. Using the same name replaces
# MAGIC the column in place.

# COMMAND ----------

t = (trades
     .withColumn("trade_ts",  F.to_timestamp("trade_ts", "yyyy-MM-dd HH:mm:ss"))
     .withColumn("quantity",  col("quantity").cast(IntegerType()))
     .withColumn("price",     col("price").cast(DecimalType(18, 6)))
     .withColumn("notional",  (col("quantity") * col("price")).cast(DecimalType(20, 6)))
     .withColumn("ingest_ts", F.current_timestamp())
     .withColumn("source_file", F.input_file_name()))

t.select("trade_id", "trade_ts", "quantity", "price", "notional").show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance note
# MAGIC Chaining 50 `withColumn` calls builds a deep plan and is measurably slower to
# MAGIC analyse. For many columns at once, prefer a single `select` or `withColumns`
# MAGIC (Spark 3.3+):

# COMMAND ----------

t2 = trades.withColumns({
    "trade_ts": F.to_timestamp("trade_ts"),
    "notional": col("quantity") * col("price"),
    "is_buy":   col("side") == "BUY",
})
t2.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Conditional logic — when / otherwise
# MAGIC
# MAGIC This is Spark's `CASE WHEN`. Omitting `otherwise` yields `NULL` for unmatched rows.

# COMMAND ----------

t = (t
     .withColumn("signed_quantity",
                 when(col("side") == "BUY",  col("quantity"))
                 .when(col("side") == "SELL", -col("quantity"))
                 .otherwise(lit(0)))
     .withColumn("trade_size_band",
                 when(col("notional") < 50_000,     lit("SMALL"))
                 .when(col("notional") < 500_000,   lit("MEDIUM"))
                 .when(col("notional") < 5_000_000, lit("LARGE"))
                 .otherwise(lit("BLOCK")))
     .withColumn("settlement_status",
                 when(col("status") == "FILLED",    lit("READY_TO_SETTLE"))
                 .when(col("status") == "PARTIAL",  lit("AWAITING_FILL"))
                 .otherwise(lit("NOT_SETTLING"))))

t.groupBy("trade_size_band").count().orderBy("count", ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Casting and safe numeric handling

# COMMAND ----------

# cast() returns NULL on failure — it never raises. Always check for the nulls it creates.
demo = spark.createDataFrame([("100",), ("abc",), (None,)], "raw_qty string")
(demo.withColumn("as_int", col("raw_qty").cast("int"))
     .withColumn("cast_failed", col("raw_qty").isNotNull() & col("raw_qty").cast("int").isNull())
     .show())

# COMMAND ----------

# MAGIC %md
# MAGIC **Money rule:** use `DecimalType(p, s)`, never `DoubleType`, for prices and
# MAGIC notionals. Floating point accumulates rounding error across billions of rows and
# MAGIC will not tie out to the accounting system.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 String functions

# COMMAND ----------

t = (t
     .withColumn("venue",     F.upper(F.trim(col("venue"))))
     .withColumn("trader_id", F.lpad(F.regexp_replace(col("trader_id"), "[^0-9]", ""), 5, "0"))
     .withColumn("venue_region",
                 when(col("venue").isin("NYSE", "NASDAQ"),  lit("AMER"))
                 .when(col("venue").isin("LSE", "XETRA"),   lit("EMEA"))
                 .otherwise(lit("APAC")))
     .withColumn("trade_key", F.concat_ws("|", col("account_id"), col("instrument_id"), col("trade_id")))
     .withColumn("trade_hash", F.sha2(col("trade_key"), 256)))

t.select("venue", "trader_id", "venue_region", "trade_key").show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC Frequently used: `upper` `lower` `trim` `ltrim` `rtrim` `lpad` `rpad` `substring`
# MAGIC `split` `concat_ws` `regexp_replace` `regexp_extract` `like` `rlike` `initcap`
# MAGIC `length` `translate` `sha2` `md5`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Date and time functions
# MAGIC
# MAGIC Business-date logic is where most investment-data bugs live: T+2 settlement,
# MAGIC weekend rolls, month-end reporting.

# COMMAND ----------

t = (t
     .withColumn("trade_date",  F.to_date("trade_ts"))
     .withColumn("trade_hour",  F.hour("trade_ts"))
     .withColumn("trade_year",  F.year("trade_ts"))
     .withColumn("trade_month", F.date_format("trade_ts", "yyyy-MM"))
     .withColumn("day_of_week", F.date_format("trade_ts", "EEEE"))
     .withColumn("is_weekend",  F.dayofweek("trade_ts").isin(1, 7))
     .withColumn("settle_date", F.date_add(F.to_date("trade_ts"), 2))          # naive T+2
     .withColumn("month_end",   F.last_day("trade_ts"))
     .withColumn("days_since_trade", F.datediff(F.current_date(), F.to_date("trade_ts"))))

t.select("trade_ts", "trade_date", "day_of_week", "is_weekend",
         "settle_date", "month_end", "days_since_trade").show(5, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Business-day T+2 (roll forward over the weekend)

# COMMAND ----------

t = t.withColumn(
    "settle_date_bd",
    when(F.dayofweek(col("settle_date")) == 7, F.date_add(col("settle_date"), 2))   # Sat -> Mon
    .when(F.dayofweek(col("settle_date")) == 1, F.date_add(col("settle_date"), 1))  # Sun -> Mon
    .otherwise(col("settle_date")))

t.select("trade_date", "settle_date", "settle_date_bd").distinct().show(8, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.6 Null handling toolkit

# COMMAND ----------

cleaned = (t
    .withColumn("quantity", F.coalesce(col("quantity"), lit(0)))          # coalesce
    .fillna({"venue": "UNKNOWN", "trader_id": "00000"})                   # per-column defaults
    .withColumn("price", when(col("price") <= 0, lit(None)).otherwise(col("price")))
    .na.drop(subset=["trade_id", "account_id"]))                          # drop only on keys

print("before:", t.count(), " after:", cleaned.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.7 Explode, arrays and structs (a taste of semi-structured data)

# COMMAND ----------

nested = (t.groupBy("account_id")
           .agg(F.collect_set("venue").alias("venues"),
                F.struct(F.count("*").alias("trade_count"),
                         F.sum("notional").alias("total_notional")).alias("metrics")))

nested.select("account_id", "venues", "metrics.trade_count", "metrics.total_notional").show(5, False)
nested.select("account_id", F.explode("venues").alias("venue")).show(8)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.8 UDFs — and why you should avoid them

# COMMAND ----------

# MAGIC %md
# MAGIC A Python UDF serialises every row to the Python process and back. It is a black box
# MAGIC to Catalyst — no predicate pushdown, no code generation. Order of preference:
# MAGIC
# MAGIC 1. Built-in function (`F.*`) — always first choice
# MAGIC 2. `F.expr()` / SQL function
# MAGIC 3. Pandas UDF (vectorised, Arrow-based) — acceptable when you truly need Python
# MAGIC 4. Python UDF — last resort

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf(DoubleType())
def commission_bps(notional: pd.Series) -> pd.Series:
    """Tiered commission in USD: 2bps under 1m, 1.5bps above."""
    return notional.where(notional < 1_000_000, notional * 0.00015).fillna(0) \
                   .where(notional >= 1_000_000, notional * 0.0002)

sample = t.select("trade_id", col("notional").cast("double").alias("notional")).limit(1000)
sample.withColumn("commission", commission_bps("notional")).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Add `gross_notional` (quantity × price) and `net_notional` (gross minus 2 bps).
# MAGIC 2. Create `trade_period` = `MORNING` (before 12:00), `AFTERNOON` (12:00–16:00),
# MAGIC    `AFTER_HOURS` (rest).
# MAGIC 3. Standardise `currency` to upper case and flag any value that is not exactly
# MAGIC    three letters.
# MAGIC 4. Build a `business_key` of account + instrument + trade date and hash it with
# MAGIC    `sha2`. Why is a hash useful in a Delta MERGE?
# MAGIC 5. Rewrite the commission Pandas UDF using only `when/otherwise`. Compare the
# MAGIC    `explain()` output of the two versions.
