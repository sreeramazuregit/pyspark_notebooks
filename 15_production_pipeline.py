# Databricks notebook source
# MAGIC %md
# MAGIC # 15 — Production Pipeline: Bronze → Silver → Gold
# MAGIC
# MAGIC **Focus:** everything from notebooks 01–14, assembled as a job you could schedule
# MAGIC
# MAGIC ### What makes a pipeline production-grade
# MAGIC * **Parameterised** — no hard-coded paths or dates
# MAGIC * **Idempotent** — safe to re-run any day, any number of times
# MAGIC * **Observable** — structured logs, run metrics, lineage
# MAGIC * **Gated** — data quality failures stop bad data reaching consumers
# MAGIC * **Recoverable** — clear restart semantics and a documented rollback

# COMMAND ----------

# MAGIC %md ## 15.1 Parameters (Databricks widgets)

# COMMAND ----------

dbutils.widgets.text("run_date",     "", "Run date (yyyy-MM-dd, blank = today)")
dbutils.widgets.dropdown("layers",   "ALL", ["ALL", "BRONZE", "SILVER", "GOLD"], "Layers to run")
dbutils.widgets.dropdown("dq_mode",  "GATE", ["GATE", "WARN", "SKIP"], "DQ behaviour")
dbutils.widgets.text("lookback_days", "3", "Restatement lookback (days)")

from datetime import date, datetime, timedelta
RUN_DATE      = dbutils.widgets.get("run_date") or str(date.today())
LAYERS        = dbutils.widgets.get("layers")
DQ_MODE       = dbutils.widgets.get("dq_mode")
LOOKBACK_DAYS = int(dbutils.widgets.get("lookback_days"))

BASE   = "/tmp/invest_platform"
RAW    = f"{BASE}/raw"
BRONZE = f"{BASE}/delta/bronze"
SILVER = f"{BASE}/delta/silver"
GOLD   = f"{BASE}/delta/gold"
AUDIT  = f"{BASE}/delta/audit"

# COMMAND ----------

# MAGIC %md ## 15.2 Framework: logging, run audit, error handling

# COMMAND ----------

import uuid, json, traceback
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, when, lit, sum as _sum
from pyspark.sql.types import *
from delta.tables import DeltaTable

RUN_ID = f"{RUN_DATE}_{uuid.uuid4().hex[:8]}"
spark.sql("CREATE DATABASE IF NOT EXISTS audit")

AUDIT_SCHEMA = ("run_id string, run_date string, step string, layer string, status string, "
                "rows_in long, rows_out long, started_ts timestamp, ended_ts timestamp, "
                "duration_sec double, message string")

_audit_rows = []

def log(msg, level="INFO"):
    print(f"[{datetime.now():%H:%M:%S}] [{level}] [{RUN_ID}] {msg}")

def step(name, layer):
    """Decorator: times a pipeline step, records an audit row, re-raises on failure."""
    def wrap(fn):
        def inner(*a, **kw):
            t0 = datetime.now()
            log(f"START {name}")
            try:
                result = fn(*a, **kw)
                rows_in, rows_out = (result if isinstance(result, tuple) else (None, result))
                t1 = datetime.now()
                _audit_rows.append((RUN_ID, RUN_DATE, name, layer, "SUCCESS",
                                    rows_in, rows_out, t0, t1, (t1-t0).total_seconds(), None))
                log(f"DONE  {name}  rows_out={rows_out}  {(t1-t0).total_seconds():.1f}s")
                return rows_out
            except Exception as e:
                t1 = datetime.now()
                _audit_rows.append((RUN_ID, RUN_DATE, name, layer, "FAILED",
                                    None, None, t0, t1, (t1-t0).total_seconds(),
                                    str(e)[:900]))
                log(f"FAIL  {name}: {e}", "ERROR")
                traceback.print_exc()
                raise
        return inner
    return wrap

def flush_audit():
    if not _audit_rows: return
    (spark.createDataFrame(_audit_rows, AUDIT_SCHEMA)
       .write.format("delta").mode("append").option("mergeSchema", "true").save(f"{AUDIT}/pipeline_runs"))
    spark.sql(f"CREATE TABLE IF NOT EXISTS audit.pipeline_runs USING DELTA LOCATION '{AUDIT}/pipeline_runs'")

log(f"Pipeline start | layers={LAYERS} | dq_mode={DQ_MODE} | lookback={LOOKBACK_DAYS}d")

# COMMAND ----------

# MAGIC %md ## 15.3 Bronze

# COMMAND ----------

TRADE_SCHEMA = StructType([StructField(c, StringType()) for c in
    ["trade_id","trade_ts","account_id","instrument_id","side","quantity","price",
     "currency","trader_id","venue","status","source_system"]])

@step("bronze_trades", "BRONZE")
def bronze_trades():
    src = (spark.read.format("csv").option("header", True)
             .option("mode", "PERMISSIVE").option("rescuedDataColumn", "_rescued_data")
             .schema(TRADE_SCHEMA).load(f"{RAW}/trades"))
    n_in = src.count()
    out = (src.withColumn("_ingest_ts", F.current_timestamp())
              .withColumn("_source_file", F.input_file_name())
              .withColumn("_run_id", lit(RUN_ID))
              .withColumn("_ingest_date", F.to_date(lit(RUN_DATE))))
    (out.write.format("delta").mode("append").option("mergeSchema", "true")
        .partitionBy("_ingest_date").save(f"{BRONZE}/trades_prod"))
    spark.sql(f"CREATE TABLE IF NOT EXISTS bronze.trades_prod USING DELTA LOCATION '{BRONZE}/trades_prod'")
    return n_in, out.count()

if LAYERS in ("ALL", "BRONZE"):
    bronze_trades()

# COMMAND ----------

# MAGIC %md ## 15.4 Silver — clean, dedupe, validate, MERGE

# COMMAND ----------

BUSINESS_COLS = ["account_id","instrument_id","side","quantity","price","currency","venue","status"]

@step("silver_trades", "SILVER")
def silver_trades():
    src = spark.table("bronze.trades_prod").filter(col("_run_id") == RUN_ID)
    n_in = src.count()

    typed = (src
        .withColumn("trade_ts",   F.to_timestamp("trade_ts", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("trade_date", F.to_date("trade_ts"))
        .withColumn("quantity",   col("quantity").cast(IntegerType()))
        .withColumn("price",      col("price").cast(DecimalType(18,6)))
        .withColumn("notional",   (col("quantity") * col("price")).cast(DecimalType(20,6)))
        .withColumn("side",       F.upper(F.trim(col("side"))))
        .withColumn("venue",      F.upper(F.trim(col("venue"))))
        .withColumn("status",     F.upper(F.trim(col("status"))))
        .withColumn("_row_hash",  F.sha2(F.concat_ws("||",
             *[F.coalesce(col(c).cast("string"), lit("~")) for c in BUSINESS_COLS]), 256)))

    deduped = (typed
        .withColumn("_rn", F.row_number().over(
            Window.partitionBy("trade_id").orderBy(col("_ingest_ts").desc())))
        .filter(col("_rn") == 1).drop("_rn"))

    validated = deduped.withColumn("_dq_errors", F.array_compact(F.array(
        when(col("trade_id").isNull(),      lit("MISSING_TRADE_ID")),
        when(col("account_id").isNull(),    lit("MISSING_ACCOUNT")),
        when(col("instrument_id").isNull(), lit("MISSING_INSTRUMENT")),
        when(col("trade_ts").isNull(),      lit("BAD_TIMESTAMP")),
        when(col("quantity").isNull() | (col("quantity") <= 0), lit("BAD_QUANTITY")),
        when(col("price").isNull() | (col("price") <= 0),       lit("BAD_PRICE")),
        when(~col("side").isin("BUY","SELL"), lit("BAD_SIDE")),
    ))).withColumn("_is_valid", F.size(col("_dq_errors")) == 0)

    good = validated.filter(col("_is_valid")).drop("_dq_errors","_is_valid","_rescued_data")
    bad  = validated.filter(~col("_is_valid"))

    if bad.count():
        (bad.write.format("delta").mode("append").option("mergeSchema","true")
            .save(f"{SILVER}/trades_quarantine_prod"))
        spark.sql(f"CREATE TABLE IF NOT EXISTS silver.trades_quarantine_prod USING DELTA LOCATION '{SILVER}/trades_quarantine_prod'")
        log(f"quarantined {bad.count():,} rows", "WARN")

    tgt = f"{SILVER}/trades_prod"
    if not DeltaTable.isDeltaTable(spark, tgt):
        good.write.format("delta").mode("overwrite").partitionBy("trade_date").save(tgt)
    else:
        (DeltaTable.forPath(spark, tgt).alias("t")
           .merge(good.alias("s"), "t.trade_id = s.trade_id AND t.trade_date = s.trade_date")
           .whenMatchedUpdateAll(condition="t._row_hash <> s._row_hash")
           .whenNotMatchedInsertAll()
           .execute())
    spark.sql(f"CREATE TABLE IF NOT EXISTS silver.trades_prod USING DELTA LOCATION '{tgt}'")
    return n_in, good.count()

if LAYERS in ("ALL", "SILVER"):
    silver_trades()

# COMMAND ----------

# MAGIC %md ## 15.5 Data quality gate

# COMMAND ----------

DQ_RULES = [
    ("P001", "trade_id IS NULL",                              "CRITICAL", 0.0),
    ("P002", "account_id IS NULL",                            "CRITICAL", 0.0),
    ("P003", "quantity IS NULL OR quantity <= 0",             "HIGH",     0.5),
    ("P004", "price IS NULL OR price <= 0",                   "CRITICAL", 0.0),
    ("P005", "side NOT IN ('BUY','SELL')",                    "CRITICAL", 0.0),
    ("P006", "abs(notional - quantity*price) > 0.01",         "HIGH",     0.1),
    ("P007", "trade_ts > current_timestamp()",                "HIGH",     0.0),
]

@step("dq_gate", "SILVER")
def dq_gate():
    df = spark.table("silver.trades_prod")
    total = df.count()
    results, blocking = [], []
    for rid, pred, sev, thr in DQ_RULES:
        failed = df.filter(pred).count()
        pct = failed/total*100 if total else 0
        status = "PASS" if pct <= thr else "FAIL"
        results.append((RUN_ID, rid, "silver.trades_prod", sev, total, failed, round(pct,4), thr, status))
        if status == "FAIL":
            log(f"{rid} {status} sev={sev} failed={failed:,} ({pct:.3f}% > {thr}%)", "WARN")
            if sev == "CRITICAL": blocking.append(rid)

    (spark.createDataFrame(results,
        "run_id string, rule_id string, table_name string, severity string, rows_checked long, "
        "rows_failed long, fail_pct double, threshold_pct double, result string")
       .withColumn("run_ts", F.current_timestamp())
       .write.format("delta").mode("append").option("mergeSchema","true").save(f"{AUDIT}/dq_results"))
    spark.sql(f"CREATE TABLE IF NOT EXISTS audit.dq_results USING DELTA LOCATION '{AUDIT}/dq_results'")

    if blocking and DQ_MODE == "GATE":
        raise Exception(f"DQ GATE FAILED — critical rules breached: {blocking}")
    if blocking:
        log(f"DQ failures present but mode={DQ_MODE} — continuing", "WARN")
    return total

if DQ_MODE != "SKIP" and LAYERS in ("ALL", "SILVER", "GOLD"):
    dq_gate()

# COMMAND ----------

# MAGIC %md ## 15.6 Gold — incremental restatement window

# COMMAND ----------

@step("gold_daily_position", "GOLD")
def gold_daily_position():
    window_start = (datetime.strptime(RUN_DATE, "%Y-%m-%d") - timedelta(days=LOOKBACK_DAYS)).date()
    log(f"restating positions from {window_start}")

    trades = spark.table("silver.trades_prod").filter(col("trade_date") >= lit(str(window_start)))
    n_in = trades.count()

    instruments = spark.table("silver.instruments").select(
        "instrument_id","ticker","asset_class","sector","country","currency")
    accounts = spark.table("bronze.accounts").select("account_id","portfolio_name","strategy","manager")
    prices = spark.table("silver.prices").select(
        "instrument_id", col("price_date").alias("trade_date"), "close_px")

    daily = (trades
        .filter(col("status").isin("FILLED","PARTIAL"))
        .withColumn("signed_qty", when(col("side")=="BUY", col("quantity")).otherwise(-col("quantity")))
        .groupBy("account_id","instrument_id","trade_date")
        .agg(_sum("signed_qty").alias("net_qty_traded"),
             _sum("notional").alias("gross_traded_notional"),
             F.count("*").alias("trade_count"),
             F.avg("price").alias("avg_exec_price")))

    w = Window.partitionBy("account_id","instrument_id").orderBy("trade_date")
    gold = (daily
        .withColumn("closing_position", _sum("net_qty_traded").over(w))
        .join(prices, ["instrument_id","trade_date"], "left")
        .join(F.broadcast(instruments), "instrument_id", "left")
        .join(F.broadcast(accounts), "account_id", "left")
        .withColumn("market_value", (col("closing_position")*col("close_px")).cast("decimal(20,4)"))
        .withColumn("position_side", when(col("closing_position")>0, lit("LONG"))
                                    .when(col("closing_position")<0, lit("SHORT")).otherwise(lit("FLAT")))
        .withColumn("_run_id", lit(RUN_ID))
        .withColumn("_updated_ts", F.current_timestamp()))

    tgt = f"{GOLD}/fact_daily_position_prod"
    if not DeltaTable.isDeltaTable(spark, tgt):
        gold.write.format("delta").mode("overwrite").partitionBy("trade_date").save(tgt)
    else:
        # Idempotent restatement: replace exactly the window we recomputed
        (gold.write.format("delta").mode("overwrite")
             .option("replaceWhere", f"trade_date >= '{window_start}'")
             .partitionBy("trade_date").save(tgt))
    spark.sql(f"CREATE TABLE IF NOT EXISTS gold.fact_daily_position_prod USING DELTA LOCATION '{tgt}'")
    return n_in, gold.count()

if LAYERS in ("ALL", "GOLD"):
    gold_daily_position()

# COMMAND ----------

# MAGIC %md
# MAGIC **`replaceWhere` is the key idempotency primitive for Gold.** It atomically replaces
# MAGIC only the partitions in the restatement window — so re-running the job for the same
# MAGIC date produces the same table, never duplicates.

# COMMAND ----------

# MAGIC %md ## 15.7 Maintenance

# COMMAND ----------

@step("maintenance", "ALL")
def maintenance():
    for t in ["silver.trades_prod", "gold.fact_daily_position_prod"]:
        try:
            spark.sql(f"OPTIMIZE {t}")
            log(f"optimized {t}")
        except Exception as e:
            log(f"optimize skipped for {t}: {str(e)[:80]}", "WARN")
    # VACUUM weekly, not every run:
    # spark.sql("VACUUM silver.trades_prod RETAIN 168 HOURS")
    return 0

if LAYERS == "ALL":
    maintenance()

# COMMAND ----------

# MAGIC %md ## 15.8 Close out the run

# COMMAND ----------

flush_audit()

spark.sql(f"""
  SELECT step, layer, status, rows_in, rows_out, round(duration_sec,1) AS secs
  FROM audit.pipeline_runs WHERE run_id = '{RUN_ID}' ORDER BY started_ts
""").show(truncate=False)

failed = spark.table("audit.pipeline_runs").filter(
    (col("run_id") == RUN_ID) & (col("status") == "FAILED")).count()

log(f"Pipeline complete | failed_steps={failed}")
dbutils.notebook.exit(json.dumps({"run_id": RUN_ID, "run_date": RUN_DATE,
                                  "status": "FAILED" if failed else "SUCCESS"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15.9 Scheduling this as a Databricks Job
# MAGIC
# MAGIC ```
# MAGIC Task 1  bronze_ingest   → notebook 15, layers=BRONZE
# MAGIC Task 2  silver_build    → depends on 1, layers=SILVER
# MAGIC Task 3  dq_gate         → depends on 2   (failure stops the DAG)
# MAGIC Task 4  gold_build      → depends on 3, layers=GOLD
# MAGIC Task 5  maintenance     → depends on 4, run weekly
# MAGIC ```
# MAGIC
# MAGIC * Job cluster, not all-purpose — cheaper and isolated.
# MAGIC * Retries: 2, with exponential backoff, on transient failures only.
# MAGIC * Timeout per task; alert on failure and on SLA miss.
# MAGIC * Parameters passed as job-level widgets so one notebook serves dev, UAT and prod.
# MAGIC * Deploy with Databricks Asset Bundles from Git — never edit notebooks in prod.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final exercises — your POC
# MAGIC
# MAGIC 1. Add an `instruments` and `prices` branch to the Bronze and Silver steps.
# MAGIC 2. Add a second Gold product (sector exposure) reusing the audit framework.
# MAGIC 3. Make the DQ rules table-driven instead of a Python list.
# MAGIC 4. Add a Slack or email notification on `dq_gate` failure.
# MAGIC 5. Run the pipeline twice for the same `run_date` and prove idempotency by
# MAGIC    comparing row counts and a checksum before and after.
# MAGIC 6. Draw the architecture diagram: sources → Bronze → Silver → Gold → consumers,
# MAGIC    with the DQ gate and audit tables marked. This is your POC deliverable.
