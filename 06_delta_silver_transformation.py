# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Silver Layer: Cleansing, Deduplication, Validation
# MAGIC
# MAGIC **Focus:** typed, conformed, deduplicated, business-ready data
# MAGIC
# MAGIC ### Silver principles
# MAGIC * Enforce real types (dates, decimals, integers).
# MAGIC * Deduplicate on the business key.
# MAGIC * Standardise codes and reference values.
# MAGIC * Validate — quarantine bad rows rather than dropping them silently.
# MAGIC * Silver is the layer analysts and data scientists actually query.

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import *

BASE, SILVER = "/tmp/invest_platform", "/tmp/invest_platform/delta/silver"
spark.sql("CREATE DATABASE IF NOT EXISTS silver")

bronze_trades = spark.table("bronze.trades")
print("bronze rows:", bronze_trades.count())

# COMMAND ----------

# MAGIC %md ## 6.1 Typed conversion with failure tracking

# COMMAND ----------

typed = (bronze_trades
    .withColumn("trade_ts",   F.to_timestamp("trade_ts", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("trade_date", F.to_date("trade_ts"))
    .withColumn("quantity",   col("quantity").cast(IntegerType()))
    .withColumn("price",      col("price").cast(DecimalType(18, 6)))
    .withColumn("notional",   (col("quantity") * col("price")).cast(DecimalType(20, 6))))

# Which casts failed? (non-null source, null target)
cast_failures = typed.filter(
    (col("quantity").isNull() & bronze_trades["quantity"].isNotNull()) |
    (col("trade_ts").isNull() & bronze_trades["trade_ts"].isNotNull()))
print("cast failures:", cast_failures.count())

# COMMAND ----------

# MAGIC %md ## 6.2 Standardise reference values

# COMMAND ----------

standardised = (typed
    .withColumn("side",     F.upper(F.trim(col("side"))))
    .withColumn("currency", F.upper(F.trim(col("currency"))))
    .withColumn("venue",    F.upper(F.trim(col("venue"))))
    .withColumn("status",   F.upper(F.trim(col("status"))))
    .withColumn("side",     when(col("side").isin("B", "BUY", "BOT"),  lit("BUY"))
                            .when(col("side").isin("S", "SELL", "SLD"), lit("SELL"))
                            .otherwise(lit("UNKNOWN")))
    .withColumn("venue_region",
                when(col("venue").isin("NYSE", "NASDAQ"), lit("AMER"))
                .when(col("venue").isin("LSE", "XETRA"),  lit("EMEA"))
                .when(col("venue").isin("TSE", "NSE"),    lit("APAC"))
                .otherwise(lit("UNKNOWN"))))

standardised.groupBy("side").count().show()
standardised.groupBy("venue_region").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Deduplication — three patterns
# MAGIC
# MAGIC | Pattern | Use when |
# MAGIC |---|---|
# MAGIC | `dropDuplicates([keys])` | any surviving row is fine |
# MAGIC | `row_number()` window | you need the **latest** by an ordering column |
# MAGIC | hash comparison | you need to detect *changed* vs *repeated* records |

# COMMAND ----------

# Pattern 2 — keep the most recently ingested version of each trade_id
w = Window.partitionBy("trade_id").orderBy(col("_ingest_ts").desc(), col("_batch_id").desc())

deduped = (standardised
    .withColumn("_rn", F.row_number().over(w))
    .filter(col("_rn") == 1)
    .drop("_rn"))

print(f"before={standardised.count():,}  after={deduped.count():,}  "
      f"removed={standardised.count()-deduped.count():,}")

# COMMAND ----------

# Pattern 3 — a row hash makes "did anything actually change?" a one-column comparison
BUSINESS_COLS = ["account_id", "instrument_id", "side", "quantity", "price", "currency", "venue", "status"]

deduped = deduped.withColumn("_row_hash", F.sha2(F.concat_ws("||", *[F.coalesce(col(c).cast("string"), lit("~")) for c in BUSINESS_COLS]), 256))
deduped.select("trade_id", "_row_hash").show(3, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.4 Validation and quarantine
# MAGIC
# MAGIC Never `filter()` bad rows into oblivion. Tag them, split them, and write the bad
# MAGIC ones to a quarantine table someone can actually investigate.

# COMMAND ----------

instruments = spark.table("bronze.instruments").select("instrument_id").distinct()
accounts    = spark.table("bronze.accounts").select("account_id").distinct()

validated = (deduped
    .withColumn("_dq_errors", F.array_compact(F.array(
        when(col("trade_id").isNull(),            lit("MISSING_TRADE_ID")),
        when(col("account_id").isNull(),          lit("MISSING_ACCOUNT_ID")),
        when(col("instrument_id").isNull(),       lit("MISSING_INSTRUMENT_ID")),
        when(col("trade_ts").isNull(),            lit("INVALID_TIMESTAMP")),
        when(col("quantity").isNull(),            lit("MISSING_QUANTITY")),
        when(col("quantity") <= 0,                lit("NON_POSITIVE_QUANTITY")),
        when(col("price").isNull() | (col("price") <= 0), lit("INVALID_PRICE")),
        when(col("side") == "UNKNOWN",            lit("UNKNOWN_SIDE")),
        when(col("trade_ts") > F.current_timestamp(), lit("FUTURE_DATED")),
    ))))

# Referential integrity via left_anti
orphan_ids = (validated.join(instruments, "instrument_id", "left_anti")
                       .select("trade_id").withColumn("_ri_error", lit("ORPHAN_INSTRUMENT")))

validated = (validated.join(orphan_ids, "trade_id", "left")
    .withColumn("_dq_errors",
                when(col("_ri_error").isNotNull(), F.array_union(col("_dq_errors"), F.array(col("_ri_error"))))
                .otherwise(col("_dq_errors")))
    .drop("_ri_error")
    .withColumn("_is_valid", F.size(col("_dq_errors")) == 0))

validated.groupBy("_is_valid").count().show()
(validated.filter(~col("_is_valid"))
          .select(F.explode("_dq_errors").alias("error"))
          .groupBy("error").count().orderBy(col("count").desc()).show(truncate=False))

# COMMAND ----------

# MAGIC %md ## 6.5 Split: clean to Silver, bad to quarantine

# COMMAND ----------

SILVER_COLS = ["trade_id", "trade_ts", "trade_date", "account_id", "instrument_id",
               "side", "quantity", "price", "notional", "currency", "trader_id",
               "venue", "venue_region", "status", "_row_hash", "_ingest_ts", "_batch_id"]

silver_trades = validated.filter(col("_is_valid")).select(*SILVER_COLS)
quarantine    = validated.filter(~col("_is_valid")).select(*SILVER_COLS, "_dq_errors")

(silver_trades.write.format("delta").mode("overwrite")
   .option("overwriteSchema", "true").partitionBy("trade_date")
   .save(f"{SILVER}/trades"))
spark.sql(f"CREATE TABLE IF NOT EXISTS silver.trades USING DELTA LOCATION '{SILVER}/trades'")

(quarantine.write.format("delta").mode("overwrite")
   .option("overwriteSchema", "true").save(f"{SILVER}/trades_quarantine"))
spark.sql(f"CREATE TABLE IF NOT EXISTS silver.trades_quarantine USING DELTA LOCATION '{SILVER}/trades_quarantine'")

print(f"silver.trades            {silver_trades.count():>8,}")
print(f"silver.trades_quarantine {quarantine.count():>8,}")

# COMMAND ----------

# MAGIC %md ## 6.6 Silver instruments and prices

# COMMAND ----------

silver_instruments = (spark.table("bronze.instruments")
    .withColumn("_rn", F.row_number().over(Window.partitionBy("instrument_id").orderBy(col("_ingest_ts").desc())))
    .filter(col("_rn") == 1).drop("_rn")
    .withColumn("lot_size", col("lot_size").cast(IntegerType()))
    .withColumn("sector",   F.coalesce(F.initcap(F.trim(col("sector"))), lit("Unclassified")))
    .withColumn("asset_class", F.upper(F.trim(col("asset_class"))))
    .select("instrument_id", "ticker", "instrument_name", "asset_class",
            "sector", "country", "currency", "lot_size", "_ingest_ts"))

silver_prices = (spark.table("bronze.prices")
    .withColumn("price_date", F.to_date("price_date"))
    .withColumn("close_px",   col("close_px").cast(DecimalType(18, 6)))
    .withColumn("open_px",    col("open_px").cast(DecimalType(18, 6)))
    .withColumn("high_px",    col("high_px").cast(DecimalType(18, 6)))
    .withColumn("low_px",     col("low_px").cast(DecimalType(18, 6)))
    .withColumn("volume",     col("volume").cast(LongType()))
    .filter(col("close_px") > 0)                       # negative prices are unusable
    .withColumn("_rn", F.row_number().over(
        Window.partitionBy("instrument_id", "price_date").orderBy(col("_ingest_ts").desc())))
    .filter(col("_rn") == 1).drop("_rn")
    .select("instrument_id", "price_date", "open_px", "high_px", "low_px", "close_px", "volume"))

for df, name in [(silver_instruments, "instruments"), (silver_prices, "prices")]:
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(f"{SILVER}/{name}"))
    spark.sql(f"CREATE TABLE IF NOT EXISTS silver.{name} USING DELTA LOCATION '{SILVER}/{name}'")
    print(f"silver.{name:<12} {df.count():>8,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Add a rule that flags a trade whose price deviates more than 20% from that day's
# MAGIC    market close.
# MAGIC 2. Report quarantine volume by `_batch_id` and error type.
# MAGIC 3. Some trades are quarantined for more than one reason. Find them.
# MAGIC 4. Write a `reprocess_quarantine()` function that re-validates quarantined rows and
# MAGIC    promotes any that now pass.
# MAGIC 5. Debate: should a trade with a null quantity be dropped, defaulted to zero, or
# MAGIC    quarantined? Justify your answer as a data steward would.
