package com.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.io.IOException;

public class WineQualityEval {
    public static void main(String[] args) throws IOException {
        // Initialize SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Evaluation")
                .master("local[*]") // Use local mode
                .getOrCreate();

        // Load the pretrained Logistic Regression model
        LogisticRegressionModel logisticRegressionModel = LogisticRegressionModel.load("file:///home/ubuntu/WineQualityPredictionModel");

        // Read and process validation dataset from local file system
        Dataset<Row> validationData = spark.read()
                .option("header", "true") // Automatically uses the first row as header
                .option("delimiter", ";")
                .csv("file:///home/ubuntu/ValidationDataset.csv");

        // Clean the column names by removing extra quotes and spaces
        String[] cleanedColumns = validationData.columns();
        for (int i = 0; i < cleanedColumns.length; i++) {
            cleanedColumns[i] = cleanedColumns[i].replaceAll("\"", "").trim(); // Remove quotes and trim spaces
        }
        validationData = validationData.toDF(cleanedColumns);

        // Cast "quality" to integer and create binary labels
        validationData = validationData.withColumn("quality", functions.col("quality").cast("int"));
        validationData = validationData.withColumn("quality", functions.when(functions.col("quality").geq(7), 1).otherwise(0));

        // Cast feature columns to DoubleType for proper processing
        String[] featureColumns = {"fixed acidity", "volatile acidity", "sulphates", "alcohol", "density"};
        for (String column : featureColumns) {
            validationData = validationData.withColumn(column, functions.col(column).cast("double"));
        }

        // Create feature vector using VectorAssembler
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureColumns)
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(validationData);

        // Apply feature scaling
        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithMean(true)
                .setWithStd(true);

        featureData = scaler.fit(featureData).transform(featureData);

        // Predict on the validation set
        Dataset<Row> predictions = logisticRegressionModel.transform(featureData);

        // Calculate F1 score
        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = f1Evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1);

        spark.stop();
    }
}
