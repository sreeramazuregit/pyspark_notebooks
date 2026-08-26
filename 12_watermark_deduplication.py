# Databricks notebook source
# MAGIC %md
# MAGIC # 12 — Late Events, Watermarks and Streaming Deduplication
# MAGIC
# MAGIC **Focus:** event time vs processing time · watermarks · exactly-once semantics
# MAGIC
# MAGIC Trades arrive out of order. A venue reconnects after an outage and replays an hour
# MAGIC of executions. Without a watermark, Spark keeps aggregation state forever and the
# MAGIC job eventually dies. With one that is too tight, you silently drop real trades.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, lit, window
from pyspark.sql.types import *
import random, datetime as dt, json

BASE       = "/tmp/invest_platform"
STREAM_IN  = f"{BASE}/stream/trades_late"
CHECKPOINT = f"{BASE}/checkpoints"
SILVER_S   = f"{BASE}/delta/silver_stream"
spark.sql("CREATE DATABASE IF NOT EXISTS silver_stream")
dbutils.fs.mkdirs(STREAM_IN)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12.1 Two clocks
# MAGIC
# MAGIC | | Meaning | Column |
# MAGIC |---|---|---|
# MAGIC | **Event time** | when the trade actually executed at the venue | `event_ts` |
# MAGIC | **Processing time** | when Spark saw it | `current_timestamp()` |
# MAGIC
# MAGIC All correct financial aggregation is done on **event time**. Processing time gives
# MAGIC different answers on every re-run, which is fatal for regulatory reporting.

# COMMAND ----------

# MAGIC %md ## 12.2 Generate events including late and duplicate arrivals

# COMMAND ----------

def emit(batch_label, n=400, late_pct=0.15, dup_pct=0.10, max_late_min=45):
    now = dt.datetime.now()
    rows = []
    for i in range(n):
        if random.random() < late_pct:
            ts = now - dt.timedelta(minutes=random.randint(10, max_late_min))
            tag = "LATE"
        else:
            ts = now - dt.timedelta(seconds=random.randint(0, 180))
            tag = "ONTIME"
        rows.append({
            "trade_id":      f"LT{batch_label}{i:06d}",
            "event_ts":      ts.strftime("%Y-%m-%d %H:%M:%S"),
            "account_id":    f"ACC{random.randint(1,40):04d}",
            "instrument_id": f"INS{random.randint(1,200):05d}",
            "side":          random.choice(["BUY","SELL"]),
            "quantity":      random.choice([100,500,1000,5000]),
            "price":         round(random.uniform(20,400),4),
            "arrival_tag":   tag,
        })
    # duplicates — the venue replayed some executions
    rows += random.sample(rows, k=int(n * dup_pct))
    dbutils.fs.put(f"{STREAM_IN}/late_{batch_label}.json",
                   "\n".join(json.dumps(r) for r in rows), overwrite=True)
    print(f"{batch_label}: {len(rows)} events ({int(n*late_pct)} late, {int(n*dup_pct)} duplicates)")

emit("a"); emit("b")

# COMMAND ----------

schema = StructType([
    StructField("trade_id", StringType()),    StructField("event_ts", StringType()),
    StructField("account_id", StringType()),  StructField("instrument_id", StringType()),
    StructField("side", StringType()),        StructField("quantity", IntegerType()),
    StructField("price", DoubleType()),       StructField("arrival_tag", StringType()),
])

events = (spark.readStream.format("json").schema(schema).load(STREAM_IN)
    .withColumn("event_ts",  F.to_timestamp("event_ts", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("ingest_ts", F.current_timestamp())
    .withColumn("notional",  col("quantity") * col("price"))
    .withColumn("lateness_min",
                (F.unix_timestamp("ingest_ts") - F.unix_timestamp("event_ts")) / 60))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12.3 The watermark
# MAGIC
# MAGIC ```python
# MAGIC df.withWatermark("event_ts", "30 minutes")
# MAGIC ```
# MAGIC
# MAGIC Spark tracks the maximum `event_ts` it has seen. The watermark is
# MAGIC `max_event_time − threshold`. Anything older is considered too late:
# MAGIC
# MAGIC * state for closed windows is **released** (memory stays bounded)
# MAGIC * rows older than the watermark are **dropped** from stateful operations
# MAGIC
# MAGIC ### Choosing the threshold
# MAGIC | Threshold | Effect |
# MAGIC |---|---|
# MAGIC | too small (1 min) | state is tiny; you drop legitimate late trades |
# MAGIC | too large (24 h) | nothing dropped; state grows, latency and cost climb |
# MAGIC | **right** | p99 of observed lateness, plus headroom |
# MAGIC
# MAGIC Measure your actual lateness distribution first — do not guess.

# COMMAND ----------

# MAGIC %md ## 12.4 Streaming deduplication

# COMMAND ----------

# MAGIC %md
# MAGIC `dropDuplicates(["trade_id"])` on a stream keeps every key ever seen in state —
# MAGIC unbounded growth. **`dropDuplicatesWithinWatermark`** (Spark 3.5+) bounds it to the
# MAGIC watermark window. Combine watermark + dedupe and you get practical exactly-once.

# COMMAND ----------

WATERMARK = "30 minutes"

deduped = (events
    .withWatermark("event_ts", WATERMARK)
    .dropDuplicates(["trade_id", "event_ts"]))     # keys + event time bound the state

q = (deduped.writeStream
    .format("delta").outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT}/late_dedup")
    .trigger(availableNow=True)
    .start(f"{SILVER_S}/trades_deduped"))
q.awaitTermination()

spark.sql(f"CREATE TABLE IF NOT EXISTS silver_stream.trades_deduped USING DELTA LOCATION '{SILVER_S}/trades_deduped'")
out = spark.table("silver_stream.trades_deduped")
print("deduplicated rows:", out.count())
print("distinct trade_id:", out.select("trade_id").distinct().count())

# COMMAND ----------

# MAGIC %md ## 12.5 Measure the lateness you actually have

# COMMAND ----------

out.select(
    F.count("*").alias("rows"),
    F.round(F.avg("lateness_min"), 2).alias("avg_late_min"),
    F.round(F.expr("percentile_approx(lateness_min, 0.50)"), 2).alias("p50"),
    F.round(F.expr("percentile_approx(lateness_min, 0.95)"), 2).alias("p95"),
    F.round(F.expr("percentile_approx(lateness_min, 0.99)"), 2).alias("p99"),
    F.round(F.max("lateness_min"), 2).alias("max_late_min"),
).show(truncate=False)

out.groupBy("arrival_tag").agg(
    F.count("*").alias("rows"),
    F.round(F.avg("lateness_min"), 1).alias("avg_late_min")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Read this output as an engineer would:** if p99 lateness is 42 minutes and your
# MAGIC watermark is 30 minutes, you are dropping roughly 1% of trades from every windowed
# MAGIC aggregate. In a trading P&L report that is a reconciliation break, not a rounding error.

# COMMAND ----------

# MAGIC %md ## 12.6 Windowed aggregation with a watermark

# COMMAND ----------

windowed = (events
    .withWatermark("event_ts", WATERMARK)
    .groupBy(window(col("event_ts"), "10 minutes", "5 minutes"), col("instrument_id"))
    .agg(F.count("*").alias("trade_count"),
         F.sum("notional").alias("notional"),
         F.avg("price").alias("avg_price"))
    .select(col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            "instrument_id", "trade_count", "notional", "avg_price"))

qw = (windowed.writeStream
    .format("delta").outputMode("append")          # append is legal because of the watermark
    .option("checkpointLocation", f"{CHECKPOINT}/late_windows")
    .trigger(availableNow=True)
    .start(f"{SILVER_S}/trade_windows"))
qw.awaitTermination()

spark.sql(f"CREATE TABLE IF NOT EXISTS silver_stream.trade_windows USING DELTA LOCATION '{SILVER_S}/trade_windows'")
spark.table("silver_stream.trade_windows").orderBy(col("window_start").desc()).show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Window types
# MAGIC
# MAGIC | Type | Call | Behaviour |
# MAGIC |---|---|---|
# MAGIC | Tumbling | `window(ts, "10 minutes")` | fixed, non-overlapping |
# MAGIC | Sliding | `window(ts, "10 minutes", "5 minutes")` | overlapping — a row lands in 2 windows |
# MAGIC | Session | `session_window(ts, "5 minutes")` | gap-based, variable length |

# COMMAND ----------

# MAGIC %md ## 12.7 Quantify what the watermark dropped

# COMMAND ----------

# Compare a strict watermark against a generous one on the same data (batch simulation)
static = spark.read.format("json").schema(schema).load(STREAM_IN) \
    .withColumn("event_ts", F.to_timestamp("event_ts", "yyyy-MM-dd HH:mm:ss"))

max_ts = static.select(F.max("event_ts")).first()[0]
for minutes in [5, 15, 30, 60, 240]:
    cutoff = F.lit(max_ts) - F.expr(f"INTERVAL {minutes} MINUTES")
    dropped = static.filter(col("event_ts") < cutoff).count()
    total   = static.count()
    print(f"watermark={minutes:>3}m  would drop {dropped:>5,} of {total:,} rows ({dropped/total*100:5.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12.8 Handling late data properly in a lakehouse
# MAGIC
# MAGIC The watermark protects the *stream*. It does not excuse losing data. The production
# MAGIC pattern is:
# MAGIC
# MAGIC 1. **Bronze append-only, no watermark** — every event lands, however late.
# MAGIC 2. **Silver stream with a watermark** — bounded state for real-time consumers.
# MAGIC 3. **A daily batch restatement** from Bronze — reprocesses the last N days so
# MAGIC    anything the watermark dropped is folded back into Gold.
# MAGIC 4. **A late-arrival monitor** on the lateness percentiles, alerting when p99 drifts
# MAGIC    toward your threshold.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Emit a file with events 3 hours old. Confirm they are dropped from the windowed
# MAGIC    aggregate but present in Bronze.
# MAGIC 2. Set the watermark to `2 minutes` and re-run. How much of the aggregate disappears?
# MAGIC 3. Replace the tumbling window with a session window keyed on `account_id`.
# MAGIC 4. Recommend a watermark for this data using the measured p99, and defend the number.
# MAGIC 5. Write the daily restatement job described in 12.8.
