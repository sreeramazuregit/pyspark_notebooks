# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Delta MERGE and Change Data Capture
# MAGIC
# MAGIC **Focus:** upserts, insert/update/delete handling, idempotent reloads
# MAGIC
# MAGIC The OMS sends us a change feed: new trades, amendments to existing ones, and
# MAGIC cancellations. `MERGE INTO` handles all three in one atomic transaction.

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, lit, when
from delta.tables import DeltaTable
import random, datetime as dt

SILVER = "/tmp/invest_platform/delta/silver"
TARGET = f"{SILVER}/trades_current"

# COMMAND ----------

# MAGIC %md ## 7.1 Build the target table

# COMMAND ----------

base = (spark.table("silver.trades")
        .withColumn("_op",         lit("I"))
        .withColumn("_valid_from", col("_ingest_ts"))
        .withColumn("_updated_ts", F.current_timestamp())
        .withColumn("_is_deleted", lit(False)))

(base.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(TARGET))
spark.sql(f"CREATE TABLE IF NOT EXISTS silver.trades_current USING DELTA LOCATION '{TARGET}'")
print("target rows:", spark.table("silver.trades_current").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.2 Simulate a CDC batch from the OMS
# MAGIC
# MAGIC Operation codes: `I` insert, `U` update (amendment), `D` delete (cancellation).

# COMMAND ----------

existing = [r.trade_id for r in spark.table("silver.trades_current").select("trade_id").limit(3000).collect()]
random.seed(7)

updates = (spark.table("silver.trades_current")
    .filter(col("trade_id").isin(random.sample(existing, 500)))
    .withColumn("quantity",  (col("quantity") * 1.10).cast("int"))     # amended quantity
    .withColumn("price",     col("price") * lit(1.02))
    .withColumn("status",    lit("FILLED"))
    .withColumn("_op",       lit("U")))

deletes = (spark.table("silver.trades_current")
    .filter(col("trade_id").isin(random.sample(existing, 100)))
    .withColumn("_op", lit("D")))

inserts = (spark.table("silver.trades_current").limit(200)
    .withColumn("trade_id", F.concat(lit("TRD9"), F.substring(col("trade_id"), 5, 7)))
    .withColumn("_op", lit("I")))

cdc_batch = (updates.unionByName(deletes).unionByName(inserts)
             .withColumn("_cdc_ts", F.current_timestamp()))

cdc_batch.groupBy("_op").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Always deduplicate the CDC batch first
# MAGIC MERGE raises `MultipleSourceRowMatchError` if two source rows match one target row.
# MAGIC Keep only the latest change per key.

# COMMAND ----------

w_latest = Window.partitionBy("trade_id").orderBy(col("_cdc_ts").desc())
cdc_clean = (cdc_batch.withColumn("_rn", F.row_number().over(w_latest))
                      .filter(col("_rn") == 1).drop("_rn"))
print(f"cdc rows={cdc_batch.count()}  after dedupe={cdc_clean.count()}")

# COMMAND ----------

# MAGIC %md ## 7.3 MERGE — Python API

# COMMAND ----------

target = DeltaTable.forPath(spark, TARGET)

UPDATE_SET = {
    "quantity":    "s.quantity",
    "price":       "s.price",
    "notional":    "s.quantity * s.price",
    "status":      "s.status",
    "venue":       "s.venue",
    "_op":         "'U'",
    "_updated_ts": "current_timestamp()",
}

(target.alias("t")
   .merge(cdc_clean.alias("s"), "t.trade_id = s.trade_id")
   .whenMatchedUpdate(condition = "s._op = 'D'",
                      set = {"_is_deleted": "true", "_op": "'D'", "_updated_ts": "current_timestamp()"})
   .whenMatchedUpdate(condition = "s._op = 'U' AND t._row_hash <> s._row_hash",
                      set = UPDATE_SET)
   .whenNotMatchedInsertAll(condition = "s._op <> 'D'")
   .execute())

print("after merge:", spark.table("silver.trades_current").count())
spark.table("silver.trades_current").groupBy("_op", "_is_deleted").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### The same thing in SQL

# COMMAND ----------

cdc_clean.createOrReplaceTempView("v_cdc")

# MAGIC %sql
# MAGIC -- MERGE INTO silver.trades_current AS t
# MAGIC -- USING v_cdc AS s ON t.trade_id = s.trade_id
# MAGIC -- WHEN MATCHED AND s._op = 'D' THEN UPDATE SET _is_deleted = true, _updated_ts = current_timestamp()
# MAGIC -- WHEN MATCHED AND s._op = 'U' AND t._row_hash <> s._row_hash THEN UPDATE SET *
# MAGIC -- WHEN NOT MATCHED AND s._op <> 'D' THEN INSERT *

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.4 Soft delete vs hard delete
# MAGIC
# MAGIC | | Soft (`_is_deleted = true`) | Hard (`whenMatchedDelete`) |
# MAGIC |---|---|---|
# MAGIC | Auditability | full history retained | gone from current version |
# MAGIC | Downstream reads | must filter `_is_deleted = false` | nothing to remember |
# MAGIC | Regulatory | preferred for trade data | required for GDPR erasure |
# MAGIC
# MAGIC In financial services, **soft delete by default**; hard delete only for a
# MAGIC documented right-to-erasure request.

# COMMAND ----------

# Hard delete form:
# .whenMatchedDelete(condition = "s._op = 'D'")

spark.sql("CREATE OR REPLACE VIEW silver.v_trades_active AS SELECT * FROM silver.trades_current WHERE _is_deleted = false")
spark.sql("SELECT count(*) AS active_trades FROM silver.v_trades_active").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.5 Idempotency — the property that lets you sleep at night
# MAGIC
# MAGIC A pipeline is idempotent if re-running the same input produces the same result.
# MAGIC MERGE on a business key gives you that for free. Re-run the merge above and the
# MAGIC counts do not change.

# COMMAND ----------

before = spark.table("silver.trades_current").count()
(DeltaTable.forPath(spark, TARGET).alias("t")
   .merge(cdc_clean.alias("s"), "t.trade_id = s.trade_id")
   .whenMatchedUpdate(condition="s._op = 'U' AND t._row_hash <> s._row_hash", set=UPDATE_SET)
   .whenNotMatchedInsertAll(condition="s._op <> 'D'")
   .execute())
after = spark.table("silver.trades_current").count()
print(f"before={before:,} after={after:,}  idempotent={before == after}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.6 MERGE performance
# MAGIC
# MAGIC * **Narrow the target.** Add a partition or range predicate to the ON clause:
# MAGIC   `t.trade_date >= (SELECT min(trade_date) FROM source)` — this lets Delta skip files.
# MAGIC * **Z-ORDER / cluster on the merge key** so matching rows sit in few files.
# MAGIC * **Enable deletion vectors** so an update rewrites metadata instead of whole files.
# MAGIC * **Compare a row hash** in the matched condition — no-op updates are the most
# MAGIC   common source of unnecessary file rewrites.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- OPTIMIZE silver.trades_current ZORDER BY (trade_id);
# MAGIC -- ALTER TABLE silver.trades_current SET TBLPROPERTIES (delta.enableDeletionVectors = true);
# MAGIC DESCRIBE HISTORY silver.trades_current;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7.7 Delta Change Data Feed — CDC *out* of your table
# MAGIC
# MAGIC Turn CDF on and Delta records every row-level change, so downstream Gold jobs can
# MAGIC read only what changed instead of rescanning the whole table.

# COMMAND ----------

spark.sql("ALTER TABLE silver.trades_current SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# After the next write:
# spark.read.format("delta").option("readChangeFeed", "true") \
#      .option("startingVersion", 3).table("silver.trades_current") \
#      .select("trade_id", "_change_type", "_commit_version").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Build a CDC batch with two changes to the same `trade_id` and run MERGE without
# MAGIC    deduplicating. Read the error carefully.
# MAGIC 2. Add a partition pruning predicate to the ON clause and compare `DESCRIBE HISTORY`
# MAGIC    metrics (`numTargetFilesScanned`).
# MAGIC 3. Implement hard delete instead of soft delete. What did you lose?
# MAGIC 4. Enable CDF, run one more merge, and read the change feed.
# MAGIC 5. Explain why comparing `_row_hash` matters when 90% of a CDC batch is unchanged.
