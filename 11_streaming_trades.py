# Databricks notebook source
# MAGIC %md
# MAGIC # 11 — Structured Streaming: Real-Time Trade Ingestion
# MAGIC
# MAGIC **Focus:** the streaming DataFrame model, sources, sinks, checkpoints, triggers
# MAGIC
# MAGIC ### The core idea
# MAGIC A stream is an **unbounded table**. Every micro-batch appends rows. The same
# MAGIC DataFrame API you already know applies — Spark handles incremental execution and
# MAGIC exactly-once state for you.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import *

BASE      = "/tmp/invest_platform"
STREAM_IN = f"{BASE}/stream/trades_in"
BRONZE_S  = f"{BASE}/delta/bronze_stream"
CHECKPOINT= f"{BASE}/checkpoints"

dbutils.fs.mkdirs(STREAM_IN)   # on Databricks; locally use os.makedirs
spark.sql("CREATE DATABASE IF NOT EXISTS bronze_stream")

# COMMAND ----------

# MAGIC %md ## 11.1 Batch vs stream — the only differences

# COMMAND ----------

# MAGIC %md
# MAGIC | | Batch | Stream |
# MAGIC |---|---|---|
# MAGIC | read | `spark.read` | `spark.readStream` |
# MAGIC | write | `df.write` | `df.writeStream` |
# MAGIC | schema | optional | **required** for file sources |
# MAGIC | state | none | checkpoint directory |
# MAGIC | completion | job ends | runs until stopped (or `availableNow`) |
# MAGIC
# MAGIC Everything in between — `filter`, `withColumn`, `join`, `groupBy` — is identical.

# COMMAND ----------

# MAGIC %md ## 11.2 Produce some trade events to stream

# COMMAND ----------

import random, datetime as dt, json

def emit_trade_file(n=500, batch_label="b1", late_pct=0.0):
    """Write one JSON file of trade events into the stream input folder."""
    now = dt.datetime.now()
    rows = []
    for i in range(n):
        late = random.random() < late_pct
        ts = now - dt.timedelta(minutes=random.randint(20, 90)) if late else now - dt.timedelta(seconds=random.randint(0, 120))
        rows.append({
            "trade_id":      f"STR{batch_label}{i:06d}",
            "event_ts":      ts.strftime("%Y-%m-%d %H:%M:%S"),
            "account_id":    f"ACC{random.randint(1,40):04d}",
            "instrument_id": f"INS{random.randint(1,500):05d}",
            "side":          random.choice(["BUY","SELL"]),
            "quantity":      random.choice([100,500,1000,5000]),
            "price":         round(random.uniform(20,400),4),
            "venue":         random.choice(["NYSE","NASDAQ","LSE","XETRA"]),
        })
    path = f"{STREAM_IN}/trades_{batch_label}.json"
    dbutils.fs.put(path, "\n".join(json.dumps(r) for r in rows), overwrite=True)
    print(f"wrote {n} events -> {path}")

emit_trade_file(500, "b1")

# COMMAND ----------

# MAGIC %md ## 11.3 Define the stream

# COMMAND ----------

event_schema = StructType([
    StructField("trade_id", StringType()),      StructField("event_ts", StringType()),
    StructField("account_id", StringType()),    StructField("instrument_id", StringType()),
    StructField("side", StringType()),          StructField("quantity", IntegerType()),
    StructField("price", DoubleType()),         StructField("venue", StringType()),
])

stream_raw = (spark.readStream
    .format("json")
    .schema(event_schema)                       # required — a stream cannot infer
    .option("maxFilesPerTrigger", 1)            # throttle for demo purposes
    .load(STREAM_IN))

print("isStreaming:", stream_raw.isStreaming)

stream_enriched = (stream_raw
    .withColumn("event_ts",     F.to_timestamp("event_ts", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("ingest_ts",    F.current_timestamp())
    .withColumn("notional",     col("quantity") * col("price"))
    .withColumn("signed_qty",   when(col("side") == "BUY", col("quantity")).otherwise(-col("quantity")))
    .withColumn("latency_secs", F.unix_timestamp("ingest_ts") - F.unix_timestamp("event_ts")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.4 Output modes
# MAGIC
# MAGIC | Mode | Writes | Valid for |
# MAGIC |---|---|---|
# MAGIC | `append` | only new rows | non-aggregated streams, or aggregations with a watermark |
# MAGIC | `update` | rows whose value changed | aggregations |
# MAGIC | `complete` | the entire result table every batch | small aggregations only |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.5 Triggers
# MAGIC
# MAGIC | Trigger | Behaviour | Use for |
# MAGIC |---|---|---|
# MAGIC | default | next batch as soon as the last finishes | lowest latency |
# MAGIC | `processingTime="30 seconds"` | fixed interval | steady, predictable cost |
# MAGIC | `availableNow=True` | process all available data, then stop | **incremental batch** — the most common production choice |
# MAGIC | `continuous="1 second"` | experimental, ~1 ms latency | rarely used |

# COMMAND ----------

# MAGIC %md ## 11.6 Start the stream

# COMMAND ----------

query = (stream_enriched.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT}/bronze_trades")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)                 # deterministic for a classroom run
    .start(f"{BRONZE_S}/trades"))

query.awaitTermination()
spark.sql(f"CREATE TABLE IF NOT EXISTS bronze_stream.trades USING DELTA LOCATION '{BRONZE_S}/trades'")
print("rows landed:", spark.table("bronze_stream.trades").count())

# COMMAND ----------

# MAGIC %md ## 11.7 Add more files — only the new ones are processed

# COMMAND ----------

emit_trade_file(300, "b2")
emit_trade_file(300, "b3")

q2 = (stream_enriched.writeStream
    .format("delta").outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT}/bronze_trades")
    .trigger(availableNow=True)
    .start(f"{BRONZE_S}/trades"))
q2.awaitTermination()

print("rows now:", spark.table("bronze_stream.trades").count())
spark.table("bronze_stream.trades").groupBy(F.substring("trade_id", 4, 2).alias("batch")).count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.8 Checkpoints — the most important concept in streaming
# MAGIC
# MAGIC The checkpoint directory stores:
# MAGIC * **offsets** — what has been read (source progress)
# MAGIC * **commits** — what has been written (sink progress)
# MAGIC * **state** — aggregation state for stateful operators
# MAGIC
# MAGIC Rules that will save you:
# MAGIC * **One checkpoint per query.** Sharing one across two queries corrupts both.
# MAGIC * **Never delete a checkpoint** to "fix" a stream — you will reprocess everything.
# MAGIC * A checkpoint is tied to the query's plan. Big logic changes may require a new
# MAGIC   checkpoint plus a controlled backfill.

# COMMAND ----------

display(dbutils.fs.ls(f"{CHECKPOINT}/bronze_trades"))

# COMMAND ----------

# MAGIC %md ## 11.9 Monitoring a running stream

# COMMAND ----------

# For a continuously running query:
# print(query.status)
# print(query.lastProgress)          # inputRowsPerSecond, processedRowsPerSecond, batchDuration
# for q in spark.streams.active: print(q.name, q.id, q.status)
# query.stop()

import json as _json
if query.lastProgress:
    p = query.lastProgress
    print("batchId          :", p.get("batchId"))
    print("numInputRows     :", p.get("numInputRows"))
    print("processedRows/s  :", p.get("processedRowsPerSecond"))
    print("durationMs       :", _json.dumps(p.get("durationMs", {})))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11.10 foreachBatch — batch operations inside a stream
# MAGIC
# MAGIC The escape hatch that unlocks MERGE, multi-sink writes and any batch-only API.

# COMMAND ----------

from delta.tables import DeltaTable

def upsert_batch(batch_df, batch_id):
    """Idempotent MERGE of a micro-batch into a Delta target."""
    target_path = f"{BRONZE_S}/trades_current"
    if not DeltaTable.isDeltaTable(spark, target_path):
        batch_df.write.format("delta").mode("overwrite").save(target_path)
        return
    (DeltaTable.forPath(spark, target_path).alias("t")
        .merge(batch_df.dropDuplicates(["trade_id"]).alias("s"), "t.trade_id = s.trade_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute())

q3 = (stream_enriched.writeStream
    .foreachBatch(upsert_batch)
    .option("checkpointLocation", f"{CHECKPOINT}/bronze_trades_merge")
    .trigger(availableNow=True)
    .start())
q3.awaitTermination()

spark.sql(f"CREATE TABLE IF NOT EXISTS bronze_stream.trades_current USING DELTA LOCATION '{BRONZE_S}/trades_current'")
print("merged table rows:", spark.table("bronze_stream.trades_current").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Change the trigger to `processingTime="10 seconds"`, emit files while it runs,
# MAGIC    and watch `lastProgress`.
# MAGIC 2. Stop the query, emit two more files, restart it. How many rows are processed?
# MAGIC 3. Point a second query at the same checkpoint. Read the error.
# MAGIC 4. Add a filter that routes trades with a null `quantity` to a quarantine table
# MAGIC    using `foreachBatch`.
# MAGIC 5. Explain when you would choose `availableNow` over a continuously running stream.
