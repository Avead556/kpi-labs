import os
from pathlib import Path

HOMEBREW_JDK = Path("/opt/homebrew/opt/openjdk@21")


def spark_session(app, cores="*", memory="2g"):
    if "JAVA_HOME" not in os.environ and HOMEBREW_JDK.exists():
        os.environ["JAVA_HOME"] = str(HOMEBREW_JDK)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession

    spark = (SparkSession.builder
             .master(f"local[{cores}]")
             .appName(app)
             .config("spark.driver.memory", memory)
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark
