# Databricks notebook source
# MAGIC %md
# MAGIC # 09 — Gold Layer: Portfolio and Risk Data Products
# MAGIC
# MAGIC **Focus:** business-facing aggregates that consumers actually query
# MAGIC
# MAGIC ### Gold principles
# MAGIC * Modelled for a consumer, not for the source system.
# MAGIC * Denormalised — joins already done, names already business-friendly.
# MAGIC * Documented, owned, and served with an SLA. This is the **data product**.

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql.functions import col, when, lit, sum as _sum

GOLD = "/tmp/invest_platform/delta/gold"
spark.sql("CREATE DATABASE IF NOT EXISTS gold")

trades      = spark.table("silver.trades")
instruments = spark.table("silver.instruments")
prices      = spark.table("silver.prices")
accounts    = spark.table("bronze.accounts").select("account_id", "portfolio_name", "strategy", "manager")

def save_gold(df, name, partition_by=None):
    w = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    if partition_by: w = w.partitionBy(partition_by)
    w.save(f"{GOLD}/{name}")
    spark.sql(f"CREATE TABLE IF NOT EXISTS gold.{name} USING DELTA LOCATION '{GOLD}/{name}'")
    print(f"gold.{name:<26} {df.count():>8,} rows")

# COMMAND ----------

# MAGIC %md ## 9.1 Daily position — the foundational data product

# COMMAND ----------

signed = trades.withColumn("signed_qty",
    when(col("side") == "BUY", col("quantity")).otherwise(-col("quantity")))

daily_activity = (signed
    .filter(col("status").isin("FILLED", "PARTIAL"))
    .groupBy("account_id", "instrument_id", "trade_date")
    .agg(_sum("signed_qty").alias("net_qty_traded"),
         _sum(when(col("side") == "BUY",  col("quantity")).otherwise(0)).alias("qty_bought"),
         _sum(when(col("side") == "SELL", col("quantity")).otherwise(0)).alias("qty_sold"),
         _sum("notional").alias("gross_traded_notional"),
         F.count("*").alias("trade_count"),
         F.avg("price").alias("avg_exec_price")))

w_run = Window.partitionBy("account_id", "instrument_id").orderBy("trade_date")

positions = (daily_activity
    .withColumn("closing_position", _sum("net_qty_traded").over(w_run))
    .withColumn("opening_position", col("closing_position") - col("net_qty_traded"))
    .join(prices.withColumnRenamed("price_date", "trade_date")
                .select("instrument_id", "trade_date", "close_px"),
          ["instrument_id", "trade_date"], "left")
    .join(F.broadcast(instruments.select("instrument_id", "ticker", "asset_class", "sector", "country", "currency")),
          "instrument_id", "left")
    .join(F.broadcast(accounts), "account_id", "left")
    .withColumn("market_value",  (col("closing_position") * col("close_px")).cast("decimal(20,4)"))
    .withColumn("position_side", when(col("closing_position") > 0, lit("LONG"))
                                 .when(col("closing_position") < 0, lit("SHORT"))
                                 .otherwise(lit("FLAT")))
    .select("trade_date", "account_id", "portfolio_name", "strategy", "manager",
            "instrument_id", "ticker", "asset_class", "sector", "country", "currency",
            "opening_position", "qty_bought", "qty_sold", "net_qty_traded", "closing_position",
            "position_side", "close_px", "market_value", "avg_exec_price",
            "trade_count", "gross_traded_notional"))

save_gold(positions, "fact_daily_position", "trade_date")
display(spark.table("gold.fact_daily_position").limit(10))

# COMMAND ----------

# MAGIC %md ## 9.2 Portfolio summary — one row per portfolio per day

# COMMAND ----------

portfolio_daily = (spark.table("gold.fact_daily_position")
    .groupBy("trade_date", "account_id", "portfolio_name", "strategy", "manager")
    .agg(
        F.countDistinct("instrument_id").alias("positions_held"),
        _sum(when(col("market_value") > 0, col("market_value")).otherwise(0)).alias("long_market_value"),
        _sum(when(col("market_value") < 0, col("market_value")).otherwise(0)).alias("short_market_value"),
        _sum("market_value").alias("net_market_value"),
        _sum(F.abs(col("market_value"))).alias("gross_market_value"),
        _sum("trade_count").alias("trades_executed"),
        _sum("gross_traded_notional").alias("turnover"),
    )
    .withColumn("net_exposure_pct",   col("net_market_value")   / col("gross_market_value"))
    .withColumn("long_short_ratio",   F.abs(col("long_market_value") / F.nullif(col("short_market_value"), lit(0)))))

save_gold(portfolio_daily, "fact_portfolio_daily", "trade_date")

# COMMAND ----------

# MAGIC %md ## 9.3 Risk data product — concentration and exposure

# COMMAND ----------

pos = spark.table("gold.fact_daily_position").filter(col("closing_position") != 0)
latest_date = pos.select(F.max("trade_date")).first()[0]
today_pos = pos.filter(col("trade_date") == latest_date)

w_pf = Window.partitionBy("account_id")

concentration = (today_pos
    .withColumn("abs_mv", F.abs(col("market_value")))
    .withColumn("portfolio_gmv", _sum("abs_mv").over(w_pf))
    .withColumn("weight_pct", col("abs_mv") / col("portfolio_gmv") * 100)
    .withColumn("rank_in_portfolio", F.row_number().over(w_pf.orderBy(col("abs_mv").desc())))
    .withColumn("concentration_flag",
        when(col("weight_pct") > 15, lit("BREACH"))
        .when(col("weight_pct") > 10, lit("WARNING"))
        .otherwise(lit("OK")))
    .select("trade_date", "account_id", "portfolio_name", "strategy", "instrument_id", "ticker",
            "sector", "asset_class", "closing_position", "market_value", "weight_pct",
            "rank_in_portfolio", "concentration_flag"))

save_gold(concentration, "fact_position_concentration")
spark.sql("""
  SELECT concentration_flag, count(*) AS positions
  FROM gold.fact_position_concentration GROUP BY concentration_flag ORDER BY 2 DESC
""").show()

# COMMAND ----------

# MAGIC %md ## 9.4 Sector and asset-class exposure

# COMMAND ----------

sector_exposure = (today_pos
    .groupBy("trade_date", "account_id", "portfolio_name", "strategy", "sector")
    .agg(_sum("market_value").alias("net_exposure"),
         _sum(F.abs(col("market_value"))).alias("gross_exposure"),
         F.countDistinct("instrument_id").alias("instruments"))
    .withColumn("pct_of_portfolio",
                col("gross_exposure") / _sum("gross_exposure").over(Window.partitionBy("account_id")) * 100))

save_gold(sector_exposure, "fact_sector_exposure")
spark.sql("""
  SELECT strategy, sector, round(sum(gross_exposure)/1e6, 1) AS gross_exposure_musd
  FROM gold.fact_sector_exposure GROUP BY strategy, sector ORDER BY 3 DESC LIMIT 15
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## 9.5 Trading activity data product — execution quality

# COMMAND ----------

execution_quality = (trades
    .filter(col("status") == "FILLED")
    .join(prices.select(col("instrument_id"), col("price_date").alias("trade_date"), "close_px"),
          ["instrument_id", "trade_date"], "left")
    .withColumn("slippage_bps",
        ((col("price") - col("close_px")) / col("close_px") * 10000).cast("decimal(12,4)"))
    .withColumn("signed_slippage_bps",
        when(col("side") == "BUY", col("slippage_bps")).otherwise(-col("slippage_bps")))
    .groupBy("trade_date", "trader_id", "venue", "venue_region")
    .agg(F.count("*").alias("trades"),
         _sum("notional").alias("notional_traded"),
         F.avg("signed_slippage_bps").alias("avg_slippage_bps"),
         F.expr("percentile_approx(signed_slippage_bps, 0.95)").alias("p95_slippage_bps"),
         F.max("notional").alias("largest_trade")))

save_gold(execution_quality, "fact_execution_quality", "trade_date")

# COMMAND ----------

# MAGIC %md ## 9.6 Serving layer: views for BI

# COMMAND ----------

spark.sql("""
CREATE OR REPLACE VIEW gold.v_portfolio_dashboard AS
SELECT p.trade_date, p.portfolio_name, p.strategy, p.manager,
       p.positions_held, p.long_market_value, p.short_market_value,
       p.net_market_value, p.gross_market_value, p.net_exposure_pct, p.turnover
FROM gold.fact_portfolio_daily p
WHERE p.trade_date >= date_sub(current_date(), 90)
""")

spark.sql("""
CREATE OR REPLACE VIEW gold.v_risk_breaches AS
SELECT * FROM gold.fact_position_concentration
WHERE concentration_flag IN ('BREACH','WARNING')
""")

spark.sql("SELECT * FROM gold.v_risk_breaches ORDER BY weight_pct DESC LIMIT 10").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9.7 Optimise Gold for read performance

# COMMAND ----------

for t in ["fact_daily_position", "fact_portfolio_daily", "fact_position_concentration"]:
    try:
        spark.sql(f"OPTIMIZE gold.{t}")
        spark.sql(f"ANALYZE TABLE gold.{t} COMPUTE STATISTICS")
        print("optimized", t)
    except Exception as e:
        print(t, "->", str(e)[:120])

# On Databricks, prefer:
# ALTER TABLE gold.fact_daily_position CLUSTER BY (account_id, trade_date);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data product contract (document this alongside the table)
# MAGIC
# MAGIC | Field | Value |
# MAGIC |---|---|
# MAGIC | Product name | Daily Position |
# MAGIC | Owner | Data Engineering — Investment Platform |
# MAGIC | Grain | account × instrument × trade_date |
# MAGIC | Refresh | daily by 07:00 local |
# MAGIC | Freshness SLA | T+1 by 07:00 |
# MAGIC | Upstream | silver.trades, silver.prices, silver.instruments |
# MAGIC | Quality gates | see notebook 10 |
# MAGIC | Consumers | Risk, Portfolio Management, Client Reporting |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC
# MAGIC 1. Add realised and unrealised P&L using average cost basis.
# MAGIC 2. Build a `dim_date` table with business-day flags and join it into the position fact.
# MAGIC 3. Add a currency dimension and produce every exposure in both local and USD.
# MAGIC 4. Create `fact_manager_performance` at manager × month grain.
# MAGIC 5. Write the data product contract for `fact_execution_quality`.
