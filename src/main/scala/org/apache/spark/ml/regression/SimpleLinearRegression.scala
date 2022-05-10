package org.apache.spark.ml.regression

import breeze.linalg.{DenseMatrix, sum}
import org.apache.spark.ml.linalg.{Vector, DenseVector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.HasMaxIter
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}


trait SimpleLinearRegressionParams extends PredictorParams with HasMaxIter {

    def setFeaturesCol(value: String): this.type = set(featuresCol, value)

    def setPredictionCol(value: String): this.type = set(predictionCol, value)

    def setLabelCol(value: String): this.type = set(labelCol, value)

    def setMaxIter(value: Int): this.type = set(maxIter, value)

    def setLearningRate(value: Double): this.type = set(learningRate, value)

    final val learningRate: Param[Double] = new DoubleParam(
        this,
        "learningRate",
        "learning rate"
    )

    def getLeaningRate: Double = $(learningRate)

    setDefault(learningRate -> 0.001)

    protected def validateAndTransformSchema(schema: StructType): StructType = {
        SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

        if (schema.fieldNames.contains($(predictionCol))) {
            SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
            schema
        } else {
            SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
        }
    }

}

class SimpleLinearRegression(
    override val uid: String
) extends Estimator[SimpleLinearRegressionModel] with SimpleLinearRegressionParams with DefaultParamsWritable {

    def this() = this(Identifiable.randomUID("linearRegression"))

    override def fit(dataset: Dataset[_]): SimpleLinearRegressionModel = {
        val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))

        val n = dataset.count()
        val epochs = getMaxIter
        val lr = getLeaningRate

        val coefficients = Vectors.zeros(numFeatures).asBreeze
        var intercept = 0.0

        // Used to convert untyped dataframes to datasets with vectors
        implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
        implicit val doubleEncoder: Encoder[Double] = ExpressionEncoder()

        val rows: Array[Vector] = dataset.select(dataset($(featuresCol)).as[Vector]).collect()

        val arr: Array[Double] = rows.flatMap(v => v.toArray)

        val x = DenseMatrix.create[Double](numFeatures, n.intValue(), arr).t
        val y = Vectors.dense(dataset.select(dataset($(labelCol)).as[Double]).collect()).asBreeze

        for (_ <- 0 until epochs) {
            val yPred = x * coefficients + intercept - y

            coefficients -= lr / n * (yPred.toDenseMatrix * x).t.toDenseVector
            intercept -= lr / n * sum(yPred)
        }

        copyValues(new SimpleLinearRegressionModel(Vectors.fromBreeze(coefficients), intercept)).setParent(this)
    }

    override def copy(extra: ParamMap): Estimator[SimpleLinearRegressionModel] = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

class SimpleLinearRegressionModel(
    override val uid: String,
    val coefficients: DenseVector,
    val intercept   : Double
) extends Model[SimpleLinearRegressionModel] with SimpleLinearRegressionParams {

    private[regression] def this(coefficients: Vector, intercept: Double) = this(
        Identifiable.randomUID("linearRegressionModel"),
        coefficients.toDense,
        intercept
    )

    override def transformSchema(schema: StructType): StructType = {
        var outputSchema = validateAndTransformSchema(schema)
        if ($(predictionCol).nonEmpty) {
            outputSchema = SchemaUtils.updateNumeric(outputSchema, $(predictionCol))
        }

        outputSchema
    }

    override def transform(dataset: Dataset[_]): DataFrame = {
        val outputSchema = transformSchema(dataset.schema, logging = true)
        val predictUDF = udf { features: Any =>
            predict(features.asInstanceOf[Vector])
        }

        dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))), outputSchema($(predictionCol)).metadata)
    }

    override def copy(extra: ParamMap): SimpleLinearRegressionModel = copyValues(
        new SimpleLinearRegressionModel(coefficients, intercept), extra
    )

    private def predict(features: Vector) = features.asBreeze.dot(coefficients.asBreeze) + intercept

}
