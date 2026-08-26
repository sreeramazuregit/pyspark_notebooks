# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Data Quality Framework
# MAGIC
# MAGIC **Focus:** null · duplicate · range · referential · business-rule checks
# MAGIC
# MAGIC A DQ framework has four parts: **rules** (declarative), an **engine** (runs them),
# MAGIC **results** (persisted, trended), and **actions** (warn, quarantine, or fail the run).

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, when, lit
from datetime import datetime
import uuid

DQ_PATH = "/tmp/invest_platform/delta/dq"
spark.sql("CREATE DATABASE IF NOT EXISTS dq")
RUN_ID = str(uuid.uuid4())[:8]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.1 The six DQ dimensions
# MAGIC
# MAGIC | Dimension | Question | Example check |
# MAGIC |---|---|---|
# MAGIC | Completeness | is anything missing? | `quantity IS NOT NULL` |
# MAGIC | Uniqueness | is anything duplicated? | one row per `trade_id` |
# MAGIC | Validity | is the value in the allowed set/range? | `side IN ('BUY','SELL')` |
# MAGIC | Accuracy | does it match reality? | exec price within 20% of market close |
# MAGIC | Consistency | do related values agree? | `notional = quantity × price` |
# MAGIC | Timeliness | did it arrive on time? | max `trade_date` ≥ yesterday |

# COMMAND ----------

# MAGIC %md ## 10.2 Declarative rule definitions

# COMMAND ----------

RULES = [
  # (rule_id, table, dimension, description, sql_predicate_for_a_FAILING_row, severity, threshold_pct)
  ("DQ001", "silver.trades", "COMPLETENESS", "trade_id must not be null",        "trade_id IS NULL",                      "CRITICAL", 0.0),
  ("DQ002", "silver.trades", "COMPLETENESS", "account_id must not be null",      "account_id IS NULL",                    "CRITICAL", 0.0),
  ("DQ003", "silver.trades", "COMPLETENESS", "quantity must not be null",        "quantity IS NULL",                      "HIGH",     0.5),
  ("DQ004", "silver.trades", "VALIDITY",     "side must be BUY or SELL",         "side NOT IN ('BUY','SELL')",            "CRITICAL", 0.0),
  ("DQ005", "silver.trades", "VALIDITY",     "quantity must be positive",        "quantity <= 0",                         "HIGH",     0.1),
  ("DQ006", "silver.trades", "VALIDITY",     "price must be positive",           "price <= 0",                            "CRITICAL", 0.0),
  ("DQ007", "silver.trades", "VALIDITY",     "currency must be a 3-letter code", "length(currency) <> 3",                 "MEDIUM",   1.0),
  ("DQ008", "silver.trades", "TIMELINESS",   "trade must not be future-dated",   "trade_ts > current_timestamp()",         "HIGH",     0.0),
  ("DQ009", "silver.trades", "CONSISTENCY",  "notional = quantity * price",      "abs(notional - quantity*price) > 0.01", "HIGH",     0.1),
  ("DQ010", "silver.prices", "VALIDITY",     "close price must be positive",     "close_px <= 0",                         "CRITICAL", 0.0),
  ("DQ011", "silver.prices", "CONSISTENCY",  "high >= low",                      "high_px < low_px",                      "HIGH",     0.0),
  ("DQ012", "silver.prices", "CONSISTENCY",  "close between low and high",       "close_px < low_px OR close_px > high_px", "MEDIUM", 5.0),
  ("DQ013", "silver.prices", "VALIDITY",     "volume must be non-negative",      "volume < 0",                            "HIGH",     0.0),
  ("DQ014", "silver.instruments","COMPLETENESS","sector must be populated",      "sector IS NULL OR sector = ''",         "MEDIUM",   3.0),
]

# COMMAND ----------

# MAGIC %md ## 10.3 The engine

# COMMAND ----------

def run_predicate_rules(rules, run_id):
    results = []
    for rule_id, table, dim, desc, predicate, severity, threshold in rules:
        df = spark.table(table)
        total  = df.count()
        failed = df.filter(predicate).count()
        pct    = (failed / total * 100) if total else 0.0
        results.append((run_id, datetime.now(), rule_id, table, dim, desc, severity,
                        total, failed, round(pct, 4), round(threshold, 4),
                        "PASS" if pct <= threshold else "FAIL"))
    schema = ("run_id string, run_ts timestamp, rule_id string, table_name string, dimension string, "
              "description string, severity string, rows_checked long, rows_failed long, "
              "fail_pct double, threshold_pct double, result string")
    return spark.createDataFrame(results, schema)

predicate_results = run_predicate_rules(RULES, RUN_ID)
predicate_results.select("rule_id", "table_name", "dimension", "rows_failed", "fail_pct", "threshold_pct", "result").show(30, False)

# COMMAND ----------

# MAGIC %md ## 10.4 Uniqueness checks

# COMMAND ----------

def uniqueness_check(rule_id, table, keys, severity="CRITICAL", threshold=0.0, run_id=RUN_ID):
    df = spark.table(table)
    total = df.count()
    dupes = (df.groupBy(*keys).count().filter(col("count") > 1)
               .agg(F.coalesce(F.sum(col("count") - 1), lit(0)).alias("extra")).first()["extra"])
    pct = (dupes / total * 100) if total else 0.0
    return (run_id, datetime.now(), rule_id, table, "UNIQUENESS",
            f"unique on {'+'.join(keys)}", severity, total, int(dupes), round(pct, 4),
            threshold, "PASS" if pct <= threshold else "FAIL")

uniq = spark.createDataFrame([
    uniqueness_check("DQ100", "silver.trades",      ["trade_id"]),
    uniqueness_check("DQ101", "silver.prices",      ["instrument_id", "price_date"]),
    uniqueness_check("DQ102", "silver.instruments", ["instrument_id"]),
], predicate_results.schema)
uniq.show(truncate=False)

# COMMAND ----------

# MAGIC %md ## 10.5 Referential integrity

# COMMAND ----------

def ri_check(rule_id, child_table, child_key, parent_table, parent_key, threshold=0.0, run_id=RUN_ID):
    child  = spark.table(child_table)
    parent = spark.table(parent_table).select(col(parent_key).alias("_pk")).distinct()
    total  = child.count()
    orphan = child.join(parent, child[child_key] == col("_pk"), "left_anti").count()
    pct = (orphan / total * 100) if total else 0.0
    return (run_id, datetime.now(), rule_id, child_table, "INTEGRITY",
            f"{child_key} must exist in {parent_table}", "HIGH", total, orphan,
            round(pct, 4), threshold, "PASS" if pct <= threshold else "FAIL")

ri = spark.createDataFrame([
    ri_check("DQ200", "silver.trades", "instrument_id", "silver.instruments", "instrument_id", threshold=0.6),
    ri_check("DQ201", "silver.trades", "account_id",    "bronze.accounts",    "account_id"),
    ri_check("DQ202", "silver.prices", "instrument_id", "silver.instruments", "instrument_id"),
], predicate_results.schema)
ri.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.6 Statistical / anomaly checks
# MAGIC
# MAGIC Rules catch what you thought of. Statistical profiling catches what you did not.

# COMMAND ----------

# Price outliers: more than 4 standard deviations from the instrument's 20-day mean
w20 = Window.partitionBy("instrument_id").orderBy("price_date").rowsBetween(-19, -1)

outliers = (spark.table("silver.prices")
    .withColumn("mean_20", F.avg("close_px").over(w20))
    .withColumn("std_20",  F.stddev("close_px").over(w20))
    .withColumn("z_score", (col("close_px") - col("mean_20")) / F.nullif(col("std_20"), lit(0)))
    .filter(F.abs(col("z_score")) > 4)
    .select("instrument_id", "price_date", "close_px", "mean_20", "z_score"))

print("price outliers:", outliers.count())
outliers.orderBy(F.abs(col("z_score")).desc()).show(10, False)

# COMMAND ----------

# Volume anomaly: daily row count vs the trailing 7-day average
volume_trend = (spark.table("silver.trades")
    .groupBy("trade_date").agg(F.count("*").alias("row_count"))
    .withColumn("avg_7d", F.avg("row_count").over(Window.orderBy("trade_date").rowsBetween(-7, -1)))
    .withColumn("variance_pct", (col("row_count") - col("avg_7d")) / col("avg_7d") * 100)
    .withColumn("volume_flag", when(F.abs(col("variance_pct")) > 50, lit("ANOMALY")).otherwise(lit("NORMAL"))))

volume_trend.orderBy(col("trade_date").desc()).show(15, False)

# COMMAND ----------

# MAGIC %md ## 10.7 Persist results and trend them

# COMMAND ----------

all_results = predicate_results.unionByName(uniq).unionByName(ri)

(all_results.write.format("delta").mode("append")
   .option("mergeSchema", "true").save(f"{DQ_PATH}/dq_results"))
spark.sql(f"CREATE TABLE IF NOT EXISTS dq.dq_results USING DELTA LOCATION '{DQ_PATH}/dq_results'")

spark.sql(f"""
  SELECT dimension, result, count(*) AS rules, sum(rows_failed) AS failing_rows
  FROM dq.dq_results WHERE run_id = '{RUN_ID}'
  GROUP BY dimension, result ORDER BY dimension
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## 10.8 The gate — decide whether the pipeline continues

# COMMAND ----------

def dq_gate(results_df, run_id, fail_on=("CRITICAL",)):
    failures = results_df.filter((col("run_id") == run_id) & (col("result") == "FAIL"))
    blocking = failures.filter(col("severity").isin(*fail_on))
    print(f"Total failures: {failures.count()}   Blocking: {blocking.count()}")
    if blocking.count() > 0:
        blocking.select("rule_id", "table_name", "description", "rows_failed", "fail_pct").show(truncate=False)
        raise Exception(f"DQ GATE FAILED — {blocking.count()} critical rule(s) breached in run {run_id}")
    if failures.count() > 0:
        print("WARNING: non-blocking DQ failures — pipeline continues, alerts raised.")
    return True

try:
    dq_gate(all_results, RUN_ID)
    print("DQ gate passed.")
except Exception as e:
    print("GATE:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.9 Scorecard for the steward

# COMMAND ----------

scorecard = (all_results.filter(col("run_id") == RUN_ID)
    .groupBy("table_name")
    .agg(F.count("*").alias("rules_run"),
         F.sum(when(col("result") == "PASS", 1).otherwise(0)).alias("rules_passed"),
         F.sum("rows_failed").alias("total_failing_rows"))
    .withColumn("score_pct", F.round(col("rules_passed") / col("rules_run") * 100, 1))
    .withColumn("grade", when(col("score_pct") >= 95, lit("A"))
                        .when(col("score_pct") >= 85, lit("B"))
                        .when(col("score_pct") >= 70, lit("C"))
                        .otherwise(lit("D"))))
scorecard.orderBy(col("score_pct").desc()).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10.10 Where this goes in production
# MAGIC
# MAGIC * **Delta constraints** — `ALTER TABLE ... ADD CONSTRAINT qty_positive CHECK (quantity > 0)`
# MAGIC   rejects bad rows at write time, before any framework runs.
# MAGIC * **DLT expectations** — `@dlt.expect_or_drop`, `@dlt.expect_or_fail` give you the
# MAGIC   same declarative rules with built-in metrics.
# MAGIC * **Great Expectations / Soda** — richer profiling and docs when you need them.
# MAGIC * **Microsoft Purview / Unity Catalog** — publish rule results as governance
# MAGIC   evidence so stewards and auditors see the same numbers you do.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ALTER TABLE silver.trades ADD CONSTRAINT positive_quantity CHECK (quantity > 0);
# MAGIC -- ALTER TABLE silver.trades ADD CONSTRAINT valid_side CHECK (side IN ('BUY','SELL'));

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Add a freshness rule: fail if the max `trade_date` is older than one business day.
# MAGIC 2. Add an accuracy rule: exec price within 20% of that day's market close.
# MAGIC 3. Make the rule list a Delta table so stewards can add rules without a code deploy.
# MAGIC 4. Build a 30-day DQ trend chart from `dq.dq_results`.
# MAGIC 5. Add a Delta CHECK constraint and try to insert a violating row. Compare that
# MAGIC    experience with the framework's.
