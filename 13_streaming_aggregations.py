# Databricks notebook source
# MAGIC %md
# MAGIC # 13 — Streaming Aggregations: Real-Time Risk
# MAGIC
# MAGIC **Focus:** windowed risk metrics · stream-static joins · stateful position tracking
# MAGIC
# MAGIC The risk desk wants intraday exposure that updates within a minute, not the
# MAGIC overnight batch. This is where streaming earns its cost.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit, window
from pyspark.sql.types import *
import random, datetime as dt, json

BASE       = "/tmp/invest_platform"
STREAM_IN  = f"{BASE}/stream/risk_events"
CHECKPOINT = f"{BASE}/checkpoints"
GOLD_S     = f"{BASE}/delta/gold_stream"
spark.sql("CREATE DATABASE IF NOT EXISTS gold_stream")
dbutils.fs.mkdirs(STREAM_IN)

# COMMAND ----------

# MAGIC %md ## 13.1 Event feed

# COMMAND ----------

def emit(label, n=800):
    now = dt.datetime.now()
    rows = [{
        "trade_id":      f"RSK{label}{i:06d}",
        "event_ts":      (now - dt.timedelta(seconds=random.randint(0, 900))).strftime("%Y-%m-%d %H:%M:%S"),
        "account_id":    f"ACC{random.randint(1,20):04d}",
        "instrument_id": f"INS{random.randint(1,100):05d}",
        "side":          random.choice(["BUY","SELL"]),
        "quantity":      random.choice([100,500,1000,5000,10000]),
        "price":         round(random.uniform(20,400),4),
        "venue":         random.choice(["NYSE","NASDAQ","LSE","XETRA"]),
        "trader_id":     f"TDR{random.randint(1,10):03d}",
    } for i in range(n)]
    dbutils.fs.put(f"{STREAM_IN}/risk_{label}.json", "\n".join(json.dumps(r) for r in rows), overwrite=True)

for lbl in ["a", "b", "c"]:
    emit(lbl)
print("events emitted")

schema = StructType([
    StructField("trade_id", StringType()),    StructField("event_ts", StringType()),
    StructField("account_id", StringType()),  StructField("instrument_id", StringType()),
    StructField("side", StringType()),        StructField("quantity", IntegerType()),
    StructField("price", DoubleType()),       StructField("venue", StringType()),
    StructField("trader_id", StringType()),
])

# COMMAND ----------

# MAGIC %md ## 13.2 Stream-static join — enrich with the security master

# COMMAND ----------

# MAGIC %md
# MAGIC A stream can join a static Delta table. Spark re-reads the static side on each
# MAGIC micro-batch, so dimension updates are picked up automatically. Broadcast it —
# MAGIC shuffling on every micro-batch is the classic streaming performance killer.
# MAGIC
# MAGIC **Supported:** stream ⋈ static inner and left outer.
# MAGIC **Not supported:** static-left-outer-stream, full outer with a stream on both sides.

# COMMAND ----------

instruments = spark.table("silver.instruments").select(
    "instrument_id", "ticker", "asset_class", "sector", "country", "currency")

events = (spark.readStream.format("json").schema(schema)
    .option("maxFilesPerTrigger", 1).load(STREAM_IN)
    .withColumn("event_ts",   F.to_timestamp("event_ts", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("notional",   col("quantity") * col("price"))
    .withColumn("signed_qty", when(col("side") == "BUY", col("quantity")).otherwise(-col("quantity")))
    .withColumn("signed_notional", when(col("side") == "BUY", col("notional")).otherwise(-col("notional")))
    .join(F.broadcast(instruments), "instrument_id", "left"))

# COMMAND ----------

# MAGIC %md ## 13.3 Rolling 5-minute exposure by sector

# COMMAND ----------

sector_risk = (events
    .withWatermark("event_ts", "15 minutes")
    .groupBy(window(col("event_ts"), "5 minutes"), col("sector"))
    .agg(F.count("*").alias("trades"),
         F.sum("notional").alias("gross_notional"),
         F.sum("signed_notional").alias("net_notional"),
         F.countDistinct("instrument_id").alias("instruments"),
         F.countDistinct("account_id").alias("accounts"),
         F.max("notional").alias("largest_trade"))
    .select(col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "sector", "trades", "gross_notional", "net_notional",
            "instruments", "accounts", "largest_trade"))

q1 = (sector_risk.writeStream
    .format("delta").outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT}/sector_risk")
    .trigger(availableNow=True)
    .start(f"{GOLD_S}/sector_risk_5min"))
q1.awaitTermination()

spark.sql(f"CREATE TABLE IF NOT EXISTS gold_stream.sector_risk_5min USING DELTA LOCATION '{GOLD_S}/sector_risk_5min'")
spark.table("gold_stream.sector_risk_5min").orderBy(col("window_start").desc()).show(12, False)

# COMMAND ----------

# MAGIC %md ## 13.4 Intraday position by account × instrument

# COMMAND ----------

intraday_position = (events
    .withWatermark("event_ts", "15 minutes")
    .groupBy(window(col("event_ts"), "15 minutes", "5 minutes"),
             col("account_id"), col("instrument_id"), col("ticker"))
    .agg(F.sum("signed_qty").alias("net_qty"),
         F.sum("notional").alias("gross_notional"),
         F.count("*").alias("trades"),
         F.avg("price").alias("vwap_approx"))
    .withColumn("position_side", when(col("net_qty") > 0, lit("LONG"))
                                .when(col("net_qty") < 0, lit("SHORT")).otherwise(lit("FLAT")))
    .select(col("window.start").alias("window_start"), col("window.end").alias("window_end"),
            "account_id", "instrument_id", "ticker", "net_qty", "position_side",
            "gross_notional", "trades", "vwap_approx"))

q2 = (intraday_position.writeStream
    .format("delta").outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT}/intraday_position")
    .trigger(availableNow=True)
    .start(f"{GOLD_S}/intraday_position"))
q2.awaitTermination()
spark.sql(f"CREATE TABLE IF NOT EXISTS gold_stream.intraday_position USING DELTA LOCATION '{GOLD_S}/intraday_position'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13.5 Real-time limit breach detection with foreachBatch
# MAGIC
# MAGIC Aggregate in the stream, then apply limit logic and write alerts in a batch context.

# COMMAND ----------

from delta.tables import DeltaTable

LIMITS = {"NOTIONAL_5MIN": 25_000_000, "SINGLE_TRADE": 4_000_000, "TRADES_5MIN": 150}

def detect_breaches(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    alerts = (batch_df
        .withColumn("breach_type",
            when(col("gross_notional") > LIMITS["NOTIONAL_5MIN"], lit("NOTIONAL_LIMIT"))
            .when(col("largest_trade")  > LIMITS["SINGLE_TRADE"],  lit("SINGLE_TRADE_LIMIT"))
            .when(col("trades")         > LIMITS["TRADES_5MIN"],   lit("VELOCITY_LIMIT")))
        .filter(col("breach_type").isNotNull())
        .withColumn("severity", when(col("gross_notional") > LIMITS["NOTIONAL_5MIN"] * 1.5, lit("CRITICAL"))
                                .otherwise(lit("WARNING")))
        .withColumn("batch_id",  lit(batch_id))
        .withColumn("alert_ts",  F.current_timestamp()))
    n = alerts.count()
    if n:
        alerts.write.format("delta").mode("append").option("mergeSchema", "true") \
              .save(f"{GOLD_S}/risk_alerts")
        print(f"[batch {batch_id}] {n} breach(es) written")
        # Production: also POST to PagerDuty / Slack / an event hub here.

q3 = (sector_risk.writeStream
    .foreachBatch(detect_breaches)
    .outputMode("update")
    .option("checkpointLocation", f"{CHECKPOINT}/risk_alerts")
    .trigger(availableNow=True)
    .start())
q3.awaitTermination()

try:
    spark.sql(f"CREATE TABLE IF NOT EXISTS gold_stream.risk_alerts USING DELTA LOCATION '{GOLD_S}/risk_alerts'")
    spark.table("gold_stream.risk_alerts").select(
        "window_start", "sector", "breach_type", "severity", "gross_notional", "trades").show(10, False)
except Exception as e:
    print("no breaches this run:", str(e)[:100])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13.6 Stateful aggregation state — what it costs
# MAGIC
# MAGIC Every open window keeps state on the executors. State size ≈
# MAGIC `distinct_keys × open_windows × row_size`. A sliding 15-minute window with a
# MAGIC 5-minute slide keeps 3 windows open per key at any time.
# MAGIC
# MAGIC **Controls:**
# MAGIC * tighten the watermark
# MAGIC * reduce grouping cardinality (aggregate at sector, not instrument)
# MAGIC * enable RocksDB state store: `spark.sql.streaming.stateStore.providerClass`
# MAGIC * enable changelog checkpointing for faster recovery

# COMMAND ----------

if q1.lastProgress:
    for op in q1.lastProgress.get("stateOperators", []):
        print("numRowsTotal      :", op.get("numRowsTotal"))
        print("numRowsUpdated    :", op.get("numRowsUpdated"))
        print("memoryUsedBytes   :", op.get("memoryUsedBytes"))
        print("numRowsDroppedByWatermark:", op.get("numRowsDroppedByWatermark"))

# spark.conf.set("spark.sql.streaming.stateStore.providerClass",
#                "com.databricks.sql.streaming.state.RocksDBStateStoreProvider")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13.7 Choosing an output mode for aggregations
# MAGIC
# MAGIC | Goal | Mode | Note |
# MAGIC |---|---|---|
# MAGIC | Immutable window results for BI | `append` | requires a watermark; results appear only after the window closes |
# MAGIC | Live dashboard updated every batch | `update` | Delta sink does not support `update` — use `foreachBatch` + MERGE |
# MAGIC | Small full snapshot | `complete` | memory sink / small tables only |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Add a VWAP that weights price by quantity instead of a plain average.
# MAGIC 2. Add a per-trader velocity metric and alert above 50 trades in 5 minutes.
# MAGIC 3. Convert the sector aggregation to `update` mode with `foreachBatch` + MERGE so a
# MAGIC    dashboard sees partial windows.
# MAGIC 4. Add a country-level concentration metric and flag any country above 40% of GMV.
# MAGIC 5. Measure state size at 5-minute vs 60-minute windows and explain the difference.
