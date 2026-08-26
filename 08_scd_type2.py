# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Slowly Changing Dimension Type 2
# MAGIC
# MAGIC **Focus:** tracking history on the security master
# MAGIC
# MAGIC An instrument's sector gets reclassified. Its currency changes after a
# MAGIC redenomination. A ticker is reassigned after a merger. If we overwrite (Type 1),
# MAGIC last year's P&L report re-runs with today's sectors and no longer ties out.
# MAGIC SCD2 keeps every version with a validity window.

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, lit, when
from delta.tables import DeltaTable

SILVER = "/tmp/invest_platform/delta/silver"
DIM    = f"{SILVER}/dim_instrument"
HIGH_DATE = "9999-12-31"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1 SCD types in one table
# MAGIC
# MAGIC | Type | Behaviour | Use for |
# MAGIC |---|---|---|
# MAGIC | 0 | never changes | original issue date |
# MAGIC | 1 | overwrite, no history | typo corrections |
# MAGIC | **2** | new row per change + validity window | sector, currency, rating |
# MAGIC | 3 | previous value in an extra column | one-step-back comparisons |
# MAGIC | 6 | 1 + 2 + 3 combined | rare, heavy |

# COMMAND ----------

# MAGIC %md ## 8.2 Initial load

# COMMAND ----------

TRACKED = ["ticker", "instrument_name", "asset_class", "sector", "country", "currency", "lot_size"]

def add_scd_cols(df, effective_date_expr=F.current_date()):
    return (df
        .withColumn("_row_hash", F.sha2(F.concat_ws("||",
                    *[F.coalesce(col(c).cast("string"), lit("~")) for c in TRACKED]), 256))
        .withColumn("effective_from", effective_date_expr)
        .withColumn("effective_to",   F.to_date(lit(HIGH_DATE)))
        .withColumn("is_current",     lit(True))
        .withColumn("version",        lit(1))
        .withColumn("_updated_ts",    F.current_timestamp()))

initial = add_scd_cols(
    spark.table("silver.instruments").select("instrument_id", *TRACKED),
    F.to_date(lit("2024-01-01")))

(initial.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(DIM))
spark.sql(f"CREATE TABLE IF NOT EXISTS silver.dim_instrument USING DELTA LOCATION '{DIM}'")
print("dim rows:", spark.table("silver.dim_instrument").count())
display(spark.table("silver.dim_instrument").limit(5))

# COMMAND ----------

# MAGIC %md ## 8.3 Simulate a new security-master snapshot with real changes

# COMMAND ----------

src = spark.table("silver.instruments").select("instrument_id", *TRACKED)

incoming = (src
    # 40 instruments reclassified into a new sector
    .withColumn("sector", when(col("instrument_id").between("INS00001", "INS00040"), lit("Technology"))
                          .otherwise(col("sector")))
    # 20 instruments redenominated to EUR
    .withColumn("currency", when(col("instrument_id").between("INS00100", "INS00120"), lit("EUR"))
                            .otherwise(col("currency")))
    # 10 lot-size changes
    .withColumn("lot_size", when(col("instrument_id").between("INS00200", "INS00210"), lit(1000))
                            .otherwise(col("lot_size"))))

# plus 5 brand-new instruments
new_rows = (src.limit(5)
    .withColumn("instrument_id", F.concat(lit("INS9"), F.substring(col("instrument_id"), 5, 4)))
    .withColumn("sector", lit("Healthcare")))

incoming = incoming.unionByName(new_rows)
print("incoming rows:", incoming.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.4 The SCD2 merge — the two-step pattern
# MAGIC
# MAGIC Delta MERGE cannot both close an old row and insert a new one for the same key in a
# MAGIC single statement. The standard solution is the **double-source trick**:
# MAGIC
# MAGIC 1. Build a source where each *changed* key appears twice: once with the real merge
# MAGIC    key (to expire the current row) and once with a `NULL` merge key (to force an
# MAGIC    insert).
# MAGIC 2. Run one MERGE keyed on `instrument_id AND is_current = true`.

# COMMAND ----------

EFFECTIVE_DATE = F.current_date()

staged = add_scd_cols(incoming, EFFECTIVE_DATE).drop("version")
current = spark.table("silver.dim_instrument").filter(col("is_current"))

# Keys whose tracked attributes actually changed
changed_keys = (staged.alias("s")
    .join(current.alias("c"), "instrument_id", "inner")
    .filter(col("s._row_hash") != col("c._row_hash"))
    .select("instrument_id"))

print("changed instruments:", changed_keys.count())

# Rows that need a NEW version inserted: changed keys + brand-new keys
to_insert = staged.join(changed_keys, "instrument_id", "left_semi") \
                  .unionByName(staged.join(current, "instrument_id", "left_anti"))

# Double-source: real key rows expire the old version; null-key rows insert the new version
merge_source = (
    staged.join(changed_keys, "instrument_id", "left_semi")
          .withColumn("_merge_key", col("instrument_id"))          # will MATCH -> expire
    .unionByName(
    to_insert.withColumn("_merge_key", lit(None).cast("string")))  # will NOT MATCH -> insert
)

print("merge source rows:", merge_source.count())

# COMMAND ----------

dim = DeltaTable.forPath(spark, DIM)

(dim.alias("t")
   .merge(merge_source.alias("s"),
          "t.instrument_id = s._merge_key AND t.is_current = true")
   .whenMatchedUpdate(
        condition = "t._row_hash <> s._row_hash",
        set = {
            "is_current":    "false",
            "effective_to":  "date_sub(s.effective_from, 1)",
            "_updated_ts":   "current_timestamp()",
        })
   .whenNotMatchedInsert(values = {
        "instrument_id":   "s.instrument_id",
        **{c: f"s.{c}" for c in TRACKED},
        "_row_hash":       "s._row_hash",
        "effective_from":  "s.effective_from",
        "effective_to":    f"to_date('{HIGH_DATE}')",
        "is_current":      "true",
        "version":         "1",
        "_updated_ts":     "current_timestamp()",
   })
   .execute())

print("dim rows after merge:", spark.table("silver.dim_instrument").count())

# COMMAND ----------

# MAGIC %md ## 8.5 Fix version numbers and inspect the history

# COMMAND ----------

w_ver = Window.partitionBy("instrument_id").orderBy("effective_from")
versioned = spark.table("silver.dim_instrument").withColumn("version", F.row_number().over(w_ver))
(versioned.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(DIM))

spark.sql("""
  SELECT instrument_id, version, sector, currency, lot_size,
         effective_from, effective_to, is_current
  FROM silver.dim_instrument
  WHERE instrument_id IN ('INS00001','INS00105','INS00205')
  ORDER BY instrument_id, version
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## 8.6 Validate the dimension

# COMMAND ----------

dim_df = spark.table("silver.dim_instrument")

checks = {
  "exactly one current row per key":
      dim_df.filter(col("is_current")).groupBy("instrument_id").count().filter(col("count") != 1).count() == 0,
  "no overlapping validity windows":
      dim_df.withColumn("next_from", F.lead("effective_from").over(w_ver))
            .filter(col("next_from").isNotNull() & (col("effective_to") >= col("next_from"))).count() == 0,
  "current rows end at high date":
      dim_df.filter(col("is_current") & (col("effective_to") != F.to_date(lit(HIGH_DATE)))).count() == 0,
  "no null business keys":
      dim_df.filter(col("instrument_id").isNull()).count() == 0,
}
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.7 Using an SCD2 dimension — point-in-time joins
# MAGIC
# MAGIC This is the whole point. A trade from March joins to the *March* version of the
# MAGIC instrument, not today's.

# COMMAND ----------

pit = spark.sql("""
  SELECT t.trade_id, t.trade_date, t.instrument_id,
         d.sector  AS sector_as_of_trade_date,
         d.currency AS currency_as_of_trade_date,
         d.version
  FROM silver.trades t
  JOIN silver.dim_instrument d
    ON t.instrument_id = d.instrument_id
   AND t.trade_date BETWEEN d.effective_from AND d.effective_to
  LIMIT 20
""")
pit.show(truncate=False)

# Compare with the naive "current only" join
spark.sql("""
  SELECT count(*) AS rows_current_join
  FROM silver.trades t JOIN silver.dim_instrument d
    ON t.instrument_id = d.instrument_id AND d.is_current = true
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Run the SCD2 merge twice with the same input. Row count must not change. Verify.
# MAGIC 2. Change `sector` for one instrument three times on three different dates. Show all
# MAGIC    four versions with their windows.
# MAGIC 3. Add a surrogate key (`instrument_sk`) using `monotonically_increasing_id()` and
# MAGIC    join facts on the surrogate instead of the natural key. What breaks if you reload?
# MAGIC 4. Implement SCD Type 1 for `instrument_name` while keeping Type 2 for `sector` in
# MAGIC    the same table.
# MAGIC 5. Write a query that returns the security master exactly as it looked on
# MAGIC    2024-06-30.
