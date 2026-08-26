# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — PySpark Fundamentals
# MAGIC
# MAGIC **Focus:** SparkSession · DataFrames · schemas · select / filter
# MAGIC
# MAGIC By the end you can explain what happens between typing `df.filter(...)` and
# MAGIC seeing rows on screen — and why nothing happens until you call an action.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 The SparkSession
# MAGIC
# MAGIC `SparkSession` is the single entry point to everything: SQL, DataFrames,
# MAGIC streaming, catalog. In Databricks a session named `spark` already exists — do
# MAGIC **not** create a new one. Outside Databricks you build one yourself.

# COMMAND ----------

# Outside Databricks (local / spark-submit) you would write:
#
# from pyspark.sql import SparkSession
# spark = (SparkSession.builder
#            .appName("investment-platform")
#            .config("spark.sql.shuffle.partitions", "200")
#            .getOrCreate())

print("Spark version :", spark.version)
print("Application   :", spark.sparkContext.appName)
print("Default parallelism :", spark.sparkContext.defaultParallelism)
print("Shuffle partitions  :", spark.conf.get("spark.sql.shuffle.partitions"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Driver and executors — the mental model
# MAGIC
# MAGIC ```
# MAGIC   YOUR CODE (Python)                 CLUSTER
# MAGIC   ┌──────────────┐        ┌───────────────────────────┐
# MAGIC   │   Driver     │  plan  │ Executor 1  │ Executor 2  │
# MAGIC   │  SparkSession│ ─────► │  core core  │  core core  │
# MAGIC   │  builds DAG  │        │  ── tasks ──│  ── tasks ──│
# MAGIC   └──────────────┘        └───────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC * The **driver** turns your DataFrame calls into a logical plan, then a physical
# MAGIC   plan, then **tasks**.
# MAGIC * **Executors** run tasks. One task processes one **partition**.
# MAGIC * `collect()`, `toPandas()`, `.count()` on a huge frame pull data **back to the
# MAGIC   driver** — the single most common cause of `OutOfMemoryError` in production.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Reading data — and why you should always declare a schema

# COMMAND ----------

BASE_PATH = "/tmp/invest_platform"
RAW_PATH  = f"{BASE_PATH}/raw"

# Option A: inferSchema — convenient, but it reads the file TWICE and often guesses wrong
df_lazy = (spark.read.option("header", True).option("inferSchema", True)
                .csv(f"{RAW_PATH}/trades"))
df_lazy.printSchema()

# COMMAND ----------

# Option B: explicit schema — one pass, deterministic types, no surprises in production
from pyspark.sql.types import (StructType, StructField, StringType,
                               IntegerType, DoubleType, TimestampType)

trade_schema = StructType([
    StructField("trade_id",      StringType(),  False),
    StructField("trade_ts",      TimestampType(), True),
    StructField("account_id",    StringType(),  True),
    StructField("instrument_id", StringType(),  True),
    StructField("side",          StringType(),  True),
    StructField("quantity",      IntegerType(), True),
    StructField("price",         DoubleType(),  True),
    StructField("currency",      StringType(),  True),
    StructField("trader_id",     StringType(),  True),
    StructField("venue",         StringType(),  True),
    StructField("status",        StringType(),  True),
    StructField("source_system", StringType(),  True),
])

trades = (spark.read
          .option("header", True)
          .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
          .schema(trade_schema)
          .csv(f"{RAW_PATH}/trades"))

trades.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC **Rule for production:** never `inferSchema` on a scheduled job. A vendor adding a
# MAGIC leading zero to an id turns a `LongType` column into `StringType` overnight and
# MAGIC your joins silently return zero rows.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Lazy evaluation: transformations vs actions

# COMMAND ----------

# TRANSFORMATIONS — build the plan, run nothing
filled = trades.filter("status = 'FILLED'")
big    = filled.filter("quantity >= 1000")
slim   = big.select("trade_id", "account_id", "instrument_id", "side", "quantity", "price")

print("Nothing has executed yet. We only have a plan.")

# COMMAND ----------

# ACTIONS — force execution
print("count  :", slim.count())
slim.show(5, truncate=False)

# Common actions: show(), count(), collect(), take(n), first(), write.save(), toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 Selecting columns — four equivalent styles

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col

trades.select("trade_id", "price").show(3)          # string names
trades.select(col("trade_id"), col("price")).show(3)  # col() — supports operators
trades.select(trades.trade_id, trades.price).show(3)  # attribute style
trades.select(F.expr("price * quantity as notional")).show(3)  # SQL expression

# COMMAND ----------

# MAGIC %md
# MAGIC Prefer `col("x")` in reusable code — it survives DataFrame reassignment and is the
# MAGIC only style that composes into expressions like `col("price") * col("quantity")`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6 Filtering
# MAGIC
# MAGIC `filter` and `where` are the same function. Combine conditions with `&`, `|`, `~`
# MAGIC and **always parenthesise** each condition — Python's operator precedence will
# MAGIC otherwise bite you.

# COMMAND ----------

us_buys = trades.filter(
    (col("side") == "BUY") &
    (col("currency") == "USD") &
    (col("quantity").isNotNull()) &
    (col("venue").isin("NYSE", "NASDAQ"))
)
print("US buy trades:", us_buys.count())

# SQL-string form is equally valid and often more readable for long predicates
us_buys_sql = trades.where("""
    side = 'BUY'
    AND currency = 'USD'
    AND quantity IS NOT NULL
    AND venue IN ('NYSE','NASDAQ')
""")
print("Same result   :", us_buys_sql.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Null handling — the trap
# MAGIC In Spark, `NULL != 'BUY'` is `NULL`, not `True`. Rows with a null column
# MAGIC **disappear** from a `!=` filter. Use `isNull()` / `isNotNull()` explicitly, or
# MAGIC `eqNullSafe()` when null should compare equal to null.

# COMMAND ----------

print("side != 'BUY'        :", trades.filter(col("quantity") != 1000).count())
print("plus explicit nulls  :", trades.filter((col("quantity") != 1000) | col("quantity").isNull()).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.7 Inspecting a DataFrame

# COMMAND ----------

trades.printSchema()
print("Columns      :", trades.columns)
print("Partitions   :", trades.rdd.getNumPartitions())
display(trades.limit(10))
trades.select("quantity", "price").summary("count", "mean", "min", "max").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.8 DataFrame ⇄ SQL — same engine, same performance

# COMMAND ----------

trades.createOrReplaceTempView("v_trades")

spark.sql("""
    SELECT venue, side, count(*) AS trade_count
    FROM v_trades
    WHERE status = 'FILLED'
    GROUP BY venue, side
    ORDER BY trade_count DESC
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Load `prices` with an **explicit schema** (no `inferSchema`).
# MAGIC 2. Return every price row where `close_px` is negative. How many are there?
# MAGIC 3. Count `trades` per `status` using the DataFrame API, then again in SQL — confirm
# MAGIC    the counts match.
# MAGIC 4. Find all trades with a null `quantity`. Which `source_system` do they come from?
# MAGIC 5. Explain in one sentence why `trades.filter(...)` printed instantly but
# MAGIC    `trades.filter(...).count()` took seconds.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key takeaways
# MAGIC * `SparkSession` is the entry point; the driver plans, executors work.
# MAGIC * Transformations are lazy; actions trigger jobs.
# MAGIC * Declare schemas in production — never infer.
# MAGIC * Nulls are not `False`; handle them explicitly.
# MAGIC * DataFrame API and SQL compile to the identical plan — choose for readability.
