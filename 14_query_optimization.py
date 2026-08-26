# Databricks notebook source
# MAGIC %md
# MAGIC # 14 — Query Optimization: Catalyst, AQE and Plans
# MAGIC
# MAGIC **Focus:** how Spark rewrites your query, and how to read the evidence
# MAGIC
# MAGIC This is the notebook that separates "I can write PySpark" from "I can run PySpark
# MAGIC in production" — and it is where most architecture interviews go deep.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, broadcast, lit
import time

trades      = spark.table("silver.trades")
instruments = spark.table("silver.instruments")
prices      = spark.table("silver.prices")

def timed(label, fn):
    t0 = time.time(); r = fn(); print(f"{label:<38} {time.time()-t0:6.2f}s   result={r}"); return r

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14.1 The Catalyst pipeline
# MAGIC
# MAGIC ```
# MAGIC   DataFrame / SQL
# MAGIC        ↓
# MAGIC   Unresolved Logical Plan     ← parsed, names not yet checked
# MAGIC        ↓  (Catalog)
# MAGIC   Analyzed Logical Plan       ← columns and types resolved
# MAGIC        ↓  (rule-based)
# MAGIC   Optimized Logical Plan      ← predicate pushdown, column pruning, constant folding
# MAGIC        ↓  (cost-based)
# MAGIC   Physical Plans → best one   ← join strategy chosen here
# MAGIC        ↓  (Tungsten)
# MAGIC   Generated Java bytecode
# MAGIC ```
# MAGIC
# MAGIC The four plans are exactly what `explain(mode="extended")` prints.

# COMMAND ----------

q = (trades.filter(col("status") == "FILLED")
           .join(instruments, "instrument_id")
           .filter(col("sector") == "Technology")
           .select("trade_id", "ticker", "quantity", "price")
           .groupBy("ticker").agg(F.sum(col("quantity") * col("price")).alias("notional")))

q.explain(mode="extended")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explain modes
# MAGIC
# MAGIC | Mode | Shows |
# MAGIC |---|---|
# MAGIC | `simple` | physical plan only (default) |
# MAGIC | `extended` | all four plans |
# MAGIC | `formatted` | physical plan plus per-operator detail — **most readable** |
# MAGIC | `cost` | plan with size and row-count statistics |
# MAGIC | `codegen` | the generated Java |

# COMMAND ----------

q.explain(mode="formatted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14.2 Catalyst rules you should be able to name
# MAGIC
# MAGIC | Rule | What it does | Evidence in the plan |
# MAGIC |---|---|---|
# MAGIC | Predicate pushdown | moves filters to the scan | `PushedFilters: [IsNotNull(status), EqualTo(status,FILLED)]` |
# MAGIC | Column pruning | reads only needed columns | short `ReadSchema` |
# MAGIC | Partition pruning | skips whole directories | `PartitionFilters` |
# MAGIC | Constant folding | `2*3` → `6` at plan time | literal in the plan |
# MAGIC | Filter combination | merges adjacent filters | one `Filter` node |
# MAGIC | Join reorder (CBO) | smallest intermediate first | different join order than written |

# COMMAND ----------

# Predicate pushdown in action — note where the filter lands
(prices.filter((col("close_px") > 100) & (col("volume") > 1_000_000))
       .select("instrument_id", "price_date", "close_px")
       .explain(mode="formatted"))

# COMMAND ----------

# MAGIC %md
# MAGIC **What blocks pushdown:** a Python UDF in the predicate, a non-deterministic function
# MAGIC (`rand()`), a filter applied after a shuffle-heavy operator, or a data source that
# MAGIC does not support it (plain CSV supports very little).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14.3 Adaptive Query Execution (AQE)
# MAGIC
# MAGIC Catalyst optimizes before execution using estimates. AQE re-optimizes **during**
# MAGIC execution using real statistics from completed stages. Three headline features:
# MAGIC
# MAGIC 1. **Coalesce shuffle partitions** — merges the 200 default partitions into as many
# MAGIC    as the data actually needs.
# MAGIC 2. **Switch join strategy** — converts sort-merge to broadcast when the real size
# MAGIC    turns out to be small.
# MAGIC 3. **Skew join handling** — splits oversized partitions into sub-partitions.

# COMMAND ----------

for k in ["spark.sql.adaptive.enabled",
          "spark.sql.adaptive.coalescePartitions.enabled",
          "spark.sql.adaptive.skewJoin.enabled",
          "spark.sql.adaptive.advisoryPartitionSizeInBytes",
          "spark.sql.adaptive.skewJoin.skewedPartitionFactor"]:
    try: print(f"{k:<58} {spark.conf.get(k)}")
    except Exception: print(f"{k:<58} (not set)")

# COMMAND ----------

def agg(): return trades.groupBy("account_id", "venue").agg(F.sum("notional")).count()

spark.conf.set("spark.sql.adaptive.enabled", "false")
timed("AQE off", agg)

spark.conf.set("spark.sql.adaptive.enabled", "true")
timed("AQE on", agg)

# With AQE on, look for AQEShuffleRead ... coalesced in the plan
trades.groupBy("account_id", "venue").agg(F.sum("notional")).explain(mode="simple")

# COMMAND ----------

# MAGIC %md ## 14.4 Join strategies — how Spark chooses

# COMMAND ----------

# MAGIC %md
# MAGIC | Strategy | When chosen | Cost |
# MAGIC |---|---|---|
# MAGIC | **BroadcastHashJoin** | one side < `autoBroadcastJoinThreshold` | no shuffle — fastest |
# MAGIC | **SortMergeJoin** | both sides large, join keys sortable | two shuffles + two sorts |
# MAGIC | **ShuffleHashJoin** | one side much smaller but too big to broadcast | one shuffle + hash build |
# MAGIC | **BroadcastNestedLoopJoin** | no equi-join condition | O(n×m) — usually a bug |
# MAGIC
# MAGIC Hints: `BROADCAST`, `MERGE`, `SHUFFLE_HASH`, `SHUFFLE_REPLICATE_NL`.

# COMMAND ----------

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
timed("SortMergeJoin", lambda: trades.join(instruments, "instrument_id").count())

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)
timed("BroadcastHashJoin", lambda: trades.join(broadcast(instruments), "instrument_id").count())

# SQL hints
spark.sql("""
  SELECT /*+ BROADCAST(i) */ i.sector, count(*) AS trades
  FROM silver.trades t JOIN silver.instruments i USING (instrument_id)
  GROUP BY i.sector
""").explain(mode="simple")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14.5 Anti-patterns and their fixes

# COMMAND ----------

# MAGIC %md
# MAGIC | Anti-pattern | Why it hurts | Fix |
# MAGIC |---|---|---|
# MAGIC | `collect()` on a large frame | pulls everything to the driver | `limit().collect()`, or write to a table |
# MAGIC | `count()` inside a loop | re-executes the whole plan each time | compute once, or cache |
# MAGIC | `select *` then use 3 columns | reads every column | project early |
# MAGIC | Python UDF for built-in logic | no codegen, no pushdown | `F.*` or `expr()` |
# MAGIC | `distinct()` on a wide frame | shuffles every column | `dropDuplicates([keys])` |
# MAGIC | `orderBy` before a write | global sort = full shuffle | sort only for the final small output |
# MAGIC | Chained unions in a loop | deep plan, slow analysis | collect frames and `reduce(unionByName)` |
# MAGIC | `cache()` everything | evicts, spills, steals memory | cache only on 3+ reuse |
# MAGIC | Small-file writes | metadata listing dominates | `OPTIMIZE`, `coalesce`, auto-compaction |

# COMMAND ----------

# Anti-pattern: repeated count in a loop
def slow():
    total = 0
    for v in ["NYSE", "NASDAQ", "LSE"]:
        total += trades.filter(col("venue") == v).count()      # 3 full scans
    return total

# Fixed: one pass
def fast():
    r = trades.filter(col("venue").isin("NYSE", "NASDAQ", "LSE")) \
              .groupBy("venue").count().collect()
    return sum(x["count"] for x in r)

timed("3 separate counts", slow)
timed("1 grouped count",   fast)

# COMMAND ----------

# MAGIC %md ## 14.6 Delta-specific optimization

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Compact small files
# MAGIC OPTIMIZE silver.trades;
# MAGIC
# MAGIC -- Co-locate rows that are filtered together (classic clustering)
# MAGIC -- OPTIMIZE silver.trades ZORDER BY (account_id, instrument_id);
# MAGIC
# MAGIC -- Modern alternative — self-tuning, no ZORDER re-runs
# MAGIC -- ALTER TABLE silver.trades CLUSTER BY (account_id, trade_date);
# MAGIC
# MAGIC -- Remove old files (default 7-day retention protects time travel and readers)
# MAGIC -- VACUUM silver.trades RETAIN 168 HOURS;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data skipping
# MAGIC Delta stores min/max statistics for the first 32 columns of every file. A filter on
# MAGIC a statistics column lets Delta skip entire files without reading them — this is why
# MAGIC column **order** matters, and why you `ZORDER`/cluster on your common filter columns.

# COMMAND ----------

spark.sql("DESCRIBE DETAIL silver.trades").select("numFiles", "sizeInBytes", "partitionColumns").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## 14.7 A practical tuning checklist

# COMMAND ----------

# MAGIC %md
# MAGIC 1. **Read the Spark UI first.** Which stage is slow? Skewed task durations? Spill?
# MAGIC 2. **Check the scan.** Is `ReadSchema` narrow? Are `PushedFilters` present?
# MAGIC 3. **Count the exchanges.** Every `Exchange` is a shuffle — can one be removed?
# MAGIC 4. **Check join strategy.** Should a side be broadcast?
# MAGIC 5. **Check partition count and skew.** `spark_partition_id()` histogram.
# MAGIC 6. **Check file layout.** Small files? Wrong partition column? Needs `OPTIMIZE`?
# MAGIC 7. **Only then** consider cache, cluster size, or config changes.
# MAGIC
# MAGIC Tuning configs before fixing layout and plan shape is treating symptoms.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Write a query that produces a `BroadcastNestedLoopJoin`. Explain why, then fix it.
# MAGIC 2. Compare `distinct()` against `dropDuplicates(["trade_id"])` on plan and runtime.
# MAGIC 3. Rewrite a Python UDF as a native expression and compare `explain(mode="codegen")`.
# MAGIC 4. Run `OPTIMIZE` and compare `numFiles` before and after.
# MAGIC 5. Set `spark.sql.shuffle.partitions` to 8 and to 800 for the same aggregation with
# MAGIC    AQE off. Explain both results.
