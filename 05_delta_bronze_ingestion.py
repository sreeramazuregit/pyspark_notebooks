# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Bronze Layer: CSV to Delta
# MAGIC
# MAGIC **Focus:** raw ingestion into Delta Lake with full audit lineage
# MAGIC
# MAGIC ### Bronze principles
# MAGIC * Land the source **as-is** — no business logic, no filtering, no dedupe.
# MAGIC * Everything is a string if it has to be; never lose a row to a cast failure.
# MAGIC * Add audit columns: when it arrived, which file it came from, which batch.
# MAGIC * Append-only. Bronze is your replay-from point when Silver logic changes.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime

BASE   = "/tmp/invest_platform"
RAW    = f"{BASE}/raw"
BRONZE = f"{BASE}/delta/bronze"
BATCH_ID = datetime.now().strftime("%Y%m%d%H%M%S")

spark.sql("CREATE DATABASE IF NOT EXISTS bronze")
print("batch_id:", BATCH_ID)

# COMMAND ----------

# MAGIC %md ## 5.1 Explicit source schemas

# COMMAND ----------

trade_schema = StructType([
    StructField("trade_id", StringType()),      StructField("trade_ts", StringType()),
    StructField("account_id", StringType()),    StructField("instrument_id", StringType()),
    StructField("side", StringType()),          StructField("quantity", StringType()),
    StructField("price", StringType()),         StructField("currency", StringType()),
    StructField("trader_id", StringType()),     StructField("venue", StringType()),
    StructField("status", StringType()),        StructField("source_system", StringType()),
])

instrument_schema = StructType([
    StructField("instrument_id", StringType()), StructField("ticker", StringType()),
    StructField("instrument_name", StringType()), StructField("asset_class", StringType()),
    StructField("sector", StringType()),        StructField("country", StringType()),
    StructField("currency", StringType()),      StructField("lot_size", StringType()),
])

price_schema = StructType([
    StructField("instrument_id", StringType()), StructField("price_date", StringType()),
    StructField("open_px", StringType()),       StructField("high_px", StringType()),
    StructField("low_px", StringType()),        StructField("close_px", StringType()),
    StructField("volume", StringType()),
])

account_schema = StructType([
    StructField("account_id", StringType()),    StructField("portfolio_name", StringType()),
    StructField("strategy", StringType()),      StructField("manager", StringType()),
    StructField("base_currency", StringType()), StructField("aum_usd", StringType()),
    StructField("opened_date", StringType()),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.2 A reusable bronze ingestion function
# MAGIC
# MAGIC Note `_rescued_data`: with `mode="PERMISSIVE"` plus `rescuedDataColumn`, any column
# MAGIC that fails to parse — or that the source added without telling you — is captured as
# MAGIC JSON instead of being silently dropped.

# COMMAND ----------

AUDIT_COLS = ["_ingest_ts", "_source_file", "_batch_id", "_source_system"]

def ingest_bronze(source_name: str, schema: StructType, table: str, source_system: str):
    df = (spark.read
          .format("csv")
          .option("header", True)
          .option("mode", "PERMISSIVE")
          .option("rescuedDataColumn", "_rescued_data")
          .schema(schema)
          .load(f"{RAW}/{source_name}"))

    bronze_df = (df
        .withColumn("_ingest_ts",     F.current_timestamp())
        .withColumn("_source_file",   F.input_file_name())
        .withColumn("_batch_id",      F.lit(BATCH_ID))
        .withColumn("_source_system", F.lit(source_system))
        .withColumn("_ingest_date",   F.current_date()))

    (bronze_df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .partitionBy("_ingest_date")
        .save(f"{BRONZE}/{table}"))

    spark.sql(f"CREATE TABLE IF NOT EXISTS bronze.{table} USING DELTA LOCATION '{BRONZE}/{table}'")
    n = bronze_df.count()
    print(f"bronze.{table:<14} +{n:>8,} rows")
    return n

# COMMAND ----------

# MAGIC %md ## 5.3 Run the ingestion

# COMMAND ----------

ingest_bronze("trades",      trade_schema,      "trades",      "OMS")
ingest_bronze("instruments", instrument_schema, "instruments", "SECURITY_MASTER")
ingest_bronze("prices",      price_schema,      "prices",      "MARKET_DATA")
ingest_bronze("accounts",    account_schema,    "accounts",    "PORTFOLIO_ACCOUNTING")

# COMMAND ----------

# MAGIC %md ## 5.4 Verify

# COMMAND ----------

for t in ["trades", "instruments", "prices", "accounts"]:
    print(f"{t:<14}", spark.table(f"bronze.{t}").count())

display(spark.table("bronze.trades").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.5 What Delta gives you that Parquet does not

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY bronze.trades;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Time travel: read the table exactly as it was at an earlier version
# MAGIC SELECT count(*) FROM bronze.trades VERSION AS OF 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL bronze.trades;

# COMMAND ----------

# MAGIC %md
# MAGIC | Capability | Parquet | Delta |
# MAGIC |---|---|---|
# MAGIC | ACID transactions | no | yes |
# MAGIC | UPDATE / DELETE / MERGE | no | yes |
# MAGIC | Schema enforcement + evolution | no | yes |
# MAGIC | Time travel | no | yes |
# MAGIC | Streaming source and sink | limited | native |
# MAGIC | File compaction (`OPTIMIZE`) | manual | built in |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.6 Incremental ingestion with Auto Loader (production pattern)
# MAGIC
# MAGIC Rerunning the batch read above re-ingests every file. Auto Loader tracks which
# MAGIC files it has already seen, in a checkpoint, and processes only new arrivals.

# COMMAND ----------

# (Databricks only — uncomment on a Databricks cluster)
#
# (spark.readStream
#    .format("cloudFiles")
#    .option("cloudFiles.format", "csv")
#    .option("cloudFiles.schemaLocation", f"{BRONZE}/_schema/trades")
#    .option("header", True)
#    .option("rescuedDataColumn", "_rescued_data")
#    .load(f"{RAW}/trades")
#    .withColumn("_ingest_ts", F.current_timestamp())
#    .withColumn("_source_file", F.col("_metadata.file_path"))
#    .writeStream
#    .format("delta")
#    .option("checkpointLocation", f"{BRONZE}/_checkpoint/trades")
#    .trigger(availableNow=True)          # process all new files, then stop
#    .toTable("bronze.trades_autoloader"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Run the ingestion twice. What happens to the row count, and why is that correct
# MAGIC    behaviour for Bronze?
# MAGIC 2. Add an extra column to one raw CSV and re-ingest. Where does it end up?
# MAGIC 3. Use `DESCRIBE HISTORY` to find your two batches, then read each with
# MAGIC    `VERSION AS OF`.
# MAGIC 4. Change `mergeSchema` to `false`, add a column, and observe the failure. Which
# MAGIC    behaviour do you want in production, and why?
# MAGIC 5. Explain why Bronze stores `quantity` as a string.
