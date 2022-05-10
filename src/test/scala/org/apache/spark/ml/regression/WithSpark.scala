package org.apache.spark.ml.regression

import org.apache.spark.sql.{SQLContext, SparkSession}

trait WithSpark {
   lazy val sparkSession: SparkSession = WithSpark.sparkSession
   lazy val sqlContext: SQLContext = WithSpark.sqlContext
}

object WithSpark {
  lazy val sparkSession: SparkSession = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val sqlContext: SQLContext = sparkSession.sqlContext
}
