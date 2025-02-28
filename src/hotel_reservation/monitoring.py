from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType


def create_or_refresh_monitoring(config, spark, workspace):
    inf_table = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`model-serving-fe_payload`")

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("no_of_adults", IntegerType(), True),
                            StructField("no_of_children", IntegerType(), True),
                            StructField("no_of_weekend_nights", IntegerType(), True),
                            StructField("no_of_week_nights", IntegerType(), True),
                            StructField("type_of_meal_plan", StringType(), True),
                            StructField("required_car_parking_space", IntegerType(), True),
                            StructField("room_type_reserved", StringType(), True),
                            StructField("lead_time", IntegerType(), True),
                            StructField("arrival_year", IntegerType(), True),
                            StructField("arrival_month", IntegerType(), True),
                            StructField("arrival_date", IntegerType(), True),
                            StructField("market_segment_type", StringType(), True),
                            StructField("repeated_guest", IntegerType(), True),
                            StructField("no_of_previous_cancellations", IntegerType(), True),
                            StructField("no_of_previous_bookings_not_canceled", IntegerType(), True),
                            StructField("avg_price_per_room", DoubleType(), True),
                            StructField("no_of_special_requests", IntegerType(), True),
                            StructField("booking_status", StringType(), True),
                            StructField("Booking_ID", StringType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        F.col("record.Booking_ID").alias("Booking_ID"),
        F.col("record.no_of_adults").alias("no_of_adults"),
        F.col("record.no_of_children").alias("no_of_children"),
        F.col("record.no_of_weekend_nights").alias("no_of_weekend_nights"),
        F.col("record.no_of_week_nights").alias("no_of_week_nights"),
        F.col("record.type_of_meal_plan").alias("type_of_meal_plan"),
        F.col("record.required_car_parking_space").alias("required_car_parking_space"),
        F.col("record.room_type_reserved").alias("room_type_reserved"),
        F.col("record.lead_time").alias("lead_time"),
        F.col("record.arrival_year").alias("arrival_year"),
        F.col("record.arrival_month").alias("arrival_month"),
        F.col("record.arrival_date").alias("arrival_date"),
        F.col("record.market_segment_type").alias("market_segment_type"),
        F.col("record.repeated_guest").alias("repeated_guest"),
        F.col("record.no_of_previous_cancellations").alias("no_of_previous_cancellations"),
        F.col("record.no_of_previous_bookings_not_canceled").alias("no_of_previous_bookings_not_canceled"),
        F.col("record.avg_price_per_room").alias("avg_price_per_room"),
        F.col("record.no_of_special_requests").alias("no_of_special_requests"),
        F.col("record.booking_status").alias("booking_status"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("hotel-reservations-fe").alias("model_name"),
    )

    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_dataset")
    inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

    df_final_with_status = (
        df_final.join(test_set.select("Booking_ID", "booking_status"), on="Booking_ID", how="left")
        .withColumnRenamed("booking_status", "booking_status_test")
        .join(inference_set_skewed.select("Booking_ID", "booking_status"), on="Booking_ID", how="left")
        .withColumnRenamed("booking_status", "booking_status_inference")
        .select(
            "*", F.coalesce(F.col("booking_status_test"), F.col("booking_status_inference")).alias("booking_status")
        )
        .drop("booking_status_test", "booking_status_inference")
        .withColumn("booking_status", F.col("booking_status").cast("double"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["booking_status", "prediction"])
    )

    hotel_features = spark.table(f"{config.catalog_name}.{config.schema_name}.hotel_reservation_features")

    df_final_with_features = df_final_with_status.join(hotel_features, on="Booking_ID", how="left")

    df_final_with_features = df_final_with_features.withColumn(
        "avg_price_per_room", F.col("avg_price_per_room").cast("double")
    )

    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config, spark, workspace):
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="booking_status",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
