# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Spark Optimization Fundamentals
# MAGIC
# MAGIC **Focus:** partitioning · broadcast joins · `explain()` · shuffle
# MAGIC
# MAGIC Performance in Spark is almost always a question of **how much data moves across
# MAGIC the network**. This notebook makes that movement visible.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, broadcast

RAW_PATH = "/tmp/invest_platform/raw"
read = lambda n: spark.read.option("header", True).option("inferSchema", True).csv(f"{RAW_PATH}/{n}")
trades, instruments, prices = read("trades"), read("instruments"), read("prices")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Partitions — the unit of parallelism
# MAGIC
# MAGIC One partition = one task = one core. Too few partitions and the cluster idles;
# MAGIC too many and scheduling overhead dominates.
# MAGIC
# MAGIC **Target: 2–4 partitions per available core, ~128 MB of data each.**

# COMMAND ----------

print("trades partitions      :", trades.rdd.getNumPartitions())
print("cluster cores          :", spark.sparkContext.defaultParallelism)
print("shuffle partitions cfg :", spark.conf.get("spark.sql.shuffle.partitions"))

# Rows per partition — skew shows up here immediately
(trades.withColumn("pid", F.spark_partition_id())
       .groupBy("pid").count().orderBy("pid").show(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### repartition vs coalesce
# MAGIC
# MAGIC | | Shuffle? | Can increase? | Result |
# MAGIC |---|---|---|---|
# MAGIC | `repartition(n)` | yes (full) | yes | even distribution |
# MAGIC | `repartition("col")` | yes (hash) | yes | co-locates by key |
# MAGIC | `coalesce(n)` | no | **no** | merges neighbours, may be uneven |
# MAGIC
# MAGIC Use `coalesce` to reduce output files cheaply; use `repartition` when you need
# MAGIC even work distribution or key co-location before a join or write.

# COMMAND ----------

print("repartition(16) :", trades.repartition(16).rdd.getNumPartitions())
print("coalesce(2)     :", trades.coalesce(2).rdd.getNumPartitions())
print("coalesce(64)    :", trades.coalesce(64).rdd.getNumPartitions(), "  <- cannot increase")
print("by key          :", trades.repartition("account_id").rdd.getNumPartitions())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Shuffle — what it costs
# MAGIC
# MAGIC A shuffle writes every partition to disk, moves it across the network, and reads it
# MAGIC back. It is triggered by: `groupBy`, `join` (sort-merge), `distinct`,
# MAGIC `repartition`, `orderBy`, and window functions.
# MAGIC
# MAGIC **Reduce shuffle by:** filtering early, selecting only needed columns, broadcasting
# MAGIC small sides, pre-partitioning by the join key, and replacing `distinct` with
# MAGIC `dropDuplicates` on specific columns.

# COMMAND ----------

# Narrow  — no shuffle:  filter, select, withColumn, union
# Wide    — shuffle:      groupBy, join, distinct, orderBy, repartition, window
narrow = trades.filter(col("status") == "FILLED").select("trade_id", "account_id")
wide   = trades.groupBy("account_id").count()

print("--- NARROW ---"); narrow.explain(mode="simple")
print("--- WIDE ---");   wide.explain(mode="simple")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Reading an explain plan
# MAGIC
# MAGIC Read **bottom-up**. Look for these operators:
# MAGIC
# MAGIC | Operator | Means |
# MAGIC |---|---|
# MAGIC | `FileScan` | reading source; check `PushedFilters` and `ReadSchema` |
# MAGIC | `Exchange hashpartitioning` | a shuffle — the expensive step |
# MAGIC | `BroadcastHashJoin` | small side broadcast — good |
# MAGIC | `SortMergeJoin` | both sides shuffled and sorted — expensive |
# MAGIC | `BroadcastNestedLoopJoin` | no join key — usually a bug |
# MAGIC | `AQEShuffleRead coalesced` | AQE merged small partitions at runtime |

# COMMAND ----------

j = trades.join(instruments, "instrument_id").groupBy("sector").count()
j.explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Broadcast joins
# MAGIC
# MAGIC If one side fits in executor memory, Spark ships a full copy to every executor and
# MAGIC skips the shuffle entirely. Automatic below
# MAGIC `spark.sql.autoBroadcastJoinThreshold` (default 10 MB) — but Spark can only guess
# MAGIC the size of a CSV, so hint explicitly.

# COMMAND ----------

print("threshold:", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))

# Explicit hint — instruments is tiny (500 rows), trades is large
fast = trades.join(broadcast(instruments), "instrument_id")
fast.explain(mode="simple")     # look for BroadcastHashJoin

# SQL equivalent:
# SELECT /*+ BROADCAST(i) */ * FROM trades t JOIN instruments i ON ...

# COMMAND ----------

import time
def timed(label, df):
    t0 = time.time(); n = df.count(); print(f"{label:<28} rows={n:>9,}  {time.time()-t0:6.2f}s")

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)      # force sort-merge
timed("sort-merge join", trades.join(instruments, "instrument_id"))

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)
timed("broadcast join",  trades.join(broadcast(instruments), "instrument_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Do not broadcast a large table.** It is materialised on the driver first, then
# MAGIC replicated to every executor. Rule of thumb: under ~100 MB, and never anything
# MAGIC that grows unbounded.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Caching — use sparingly

# COMMAND ----------

from pyspark import StorageLevel

hot = trades.filter(col("status") == "FILLED").select("account_id", "instrument_id", "quantity", "price")
hot.persist(StorageLevel.MEMORY_AND_DISK)
hot.count()                                    # materialise the cache

timed("cached agg 1", hot.groupBy("account_id").count())
timed("cached agg 2", hot.groupBy("instrument_id").count())

hot.unpersist()                                # ALWAYS release it

# COMMAND ----------

# MAGIC %md
# MAGIC Cache only when a DataFrame is reused **three or more times** and the recompute is
# MAGIC expensive. On Delta with disk caching enabled, explicit `cache()` is often
# MAGIC unnecessary and just steals executor memory.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.6 Data skew
# MAGIC
# MAGIC One key with far more rows than the rest means one task runs long after the others
# MAGIC finish. Classic in trading data: one mega-account, or a `UNKNOWN` fallback id.

# COMMAND ----------

skew = (trades.groupBy("instrument_id").count()
              .orderBy(col("count").desc()))
skew.show(5)
stats = skew.select(F.avg("count").alias("avg"), F.max("count").alias("max")).first()
print(f"avg rows/key={stats['avg']:.0f}  max={stats['max']}  skew factor={stats['max']/stats['avg']:.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fixes for skew
# MAGIC 1. **AQE skew join** — `spark.sql.adaptive.skewJoin.enabled=true` (on by default in
# MAGIC    Databricks) splits oversized partitions automatically.
# MAGIC 2. **Salting** — add a random suffix to the hot key, join, then aggregate away the salt.
# MAGIC 3. **Broadcast the small side** — no shuffle means no skew.
# MAGIC 4. **Filter out sentinel keys** (`UNKNOWN`, `-1`) before the join.

# COMMAND ----------

# Salting sketch
SALT = 16
salted_left  = trades.withColumn("salt", (F.rand() * SALT).cast("int"))
salted_right = (instruments
    .withColumn("salt", F.explode(F.array([F.lit(i) for i in range(SALT)]))))
salted = salted_left.join(salted_right, ["instrument_id", "salt"]).drop("salt")
print("salted join rows:", salted.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.7 File layout matters more than code

# COMMAND ----------

OUT = "/tmp/invest_platform/opt_demo"

# Partitioned write — enables partition pruning on the filter column
(prices.withColumn("price_year", F.year("price_date"))
       .write.mode("overwrite").partitionBy("price_year").parquet(f"{OUT}/prices_partitioned"))

pruned = spark.read.parquet(f"{OUT}/prices_partitioned").filter(col("price_year") == 2026)
pruned.explain(mode="simple")      # PartitionFilters should appear in the FileScan

# COMMAND ----------

# MAGIC %md
# MAGIC **Partition column rules**
# MAGIC * Low cardinality (date, region, asset class) — never `trade_id` or a timestamp.
# MAGIC * Target ≥ 1 GB per partition directory.
# MAGIC * More than ~10,000 partitions creates the small-file problem: metadata listing
# MAGIC   dominates and every query slows down.
# MAGIC * On Databricks, prefer **liquid clustering** over `partitionBy` for new tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Run the trades ⋈ instruments join with and without `broadcast()`. Record both
# MAGIC    times and both plan shapes.
# MAGIC 2. Write `prices` partitioned by `price_date` (high cardinality). Count the output
# MAGIC    files. Explain why this is a bad idea.
# MAGIC 3. Find the most skewed `account_id` in trades. Compute the skew factor.
# MAGIC 4. Turn AQE off (`spark.sql.adaptive.enabled=false`), re-run an aggregation, and
# MAGIC    compare the number of output partitions.
# MAGIC 5. Take a query that reads 12 columns and uses 3. Rewrite it with an early `select`
# MAGIC    and compare `ReadSchema` in the plan.
