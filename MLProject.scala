
// Machine Learning Project- Mushroom Classification
// Big Data Analytics-COMP6130
//*********************************************************************************************************************
                                  /** Loading necessary Libraries & Packages */
//*********************************************************************************************************************
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.sql.functions.{col, countDistinct, lit, monotonicallyIncreasingId, struct, sum}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

//*********************************************************************************************************************
                                        /** SPARK PROJECT SET UP */
//*********************************************************************************************************************

object MLProject {

  case class MLSchema(id: Int, label: Int, cap_shape: String, cap_surface: String, cap_color: String, bruises: String, odor: String, gill_attachment: String, gill_spacing: String, gill_size: String,gill_color: String, stalk_shape: String, stalk_root: String, stalk_surface_above_ring: String, stalk_surface_below_ring: String, stalk_color_above_ring: String, stalk_color_below_ring: String, veil_type: String,  veil_color: String, ring_number: String, ring_type: String, spore_print_color: String, population: String, habitat: String)

 /** Our main function where the action happens */

   def main(args: Array[String]): Unit = {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Make a session
    val spark = SparkSession
      .builder
      .appName("MLProject")
      .master("local[*]")
      .getOrCreate()

//*********************************************************************************************************************
                                               /** DATA EXTRACTION */
//*********************************************************************************************************************

    // Code block loads in both training ("Frame") and test ("Test") data sets and cache both In-Memory
    val Frame = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("C:/Users/Dr. Elombe Calvert/IdeaProjects/Hello World/data/mushroom_train.csv")

     val Test = spark.read.format("csv")
       .option("header", "true")
       .option("inferSchema", "true")
       .load("C:/Users/Dr. Elombe Calvert/IdeaProjects/Hello World/data/mushroom_test.csv")

    // cache DataFrames In-Memory
    Frame.cache()
    Test.cache()
//*********************************************************************************************************************
                                         /** DATA EXPLORATION */
//*********************************************************************************************************************
    // show basic statistics of the DataFrame
    // summary statistics revealed that each column contains the same amount of data points, which is 6499.
       Frame.describe().show()

    // returns the sum of null values for each column in the data set
    // No null values were found within the Dataset
       Frame.select(Frame.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*).show

    // Count the number of distinct data points for all columns
             Frame.agg(
                countDistinct("class"),
                countDistinct("cap-shape"),
                countDistinct("cap-color"),
                countDistinct("cap-surface"),
                countDistinct("bruises"),
                countDistinct("veil-type"),
                countDistinct("ring-number"),
                countDistinct("gill-attachment"),
                countDistinct("gill-spacing"),
                countDistinct("gill-size"),
                countDistinct("gill-color"),
                countDistinct("stalk-shape"),
                countDistinct("stalk-root"),
                countDistinct("stalk-surface-above-ring"),
                countDistinct("stalk-surface-below-ring"),
                countDistinct("stalk-color-above-ring"),
                countDistinct("stalk-color-below-ring"),
                countDistinct("veil-color"),
                countDistinct("ring-type"),
                countDistinct("spore-print-color"),
                countDistinct("population"),
                countDistinct("habitat")
             ).show()

    // Shows distinct values for each column. This was done for each column
    // Based on data dictionary provided by the UC Irvine ML Repo website: https://archive.ics.uci.edu/ml/datasets/mushroom
    // No noise was found within the data

        Frame.select("class").distinct().show()
        Frame.select("cap-shape").distinct().show()
        Frame.select("cap-color").distinct().show()
        Frame.select("cap-surface").distinct().show()
        Frame.select("bruises").distinct().show()
        Frame.select("veil-type").distinct().show()
        Frame.select("ring-number").distinct().show()
        Frame.select("gill-attachment").distinct().show()
        Frame.select("gill-spacing").distinct().show()
        Frame.select("gill-size").distinct().show()
        Frame.select("gill-color").distinct().show()
        Frame.select("stalk-shape").distinct().show()
        Frame.select("stalk-root").distinct().show()
        Frame.select("stalk-surface-above-ring").distinct().show()
        Frame.select("stalk-surface-below-ring").distinct().show()
        Frame.select("stalk-color-above-ring").distinct().show()
        Frame.select("stalk-color-below-ring").distinct().show()
        Frame.select("veil-color").distinct().show()
        Frame.select("ring-type").distinct().show()
        Frame.select("spore-print-color").distinct().show()
        Frame.select("population").distinct().show()
        Frame.select("veil-color").distinct().show()
        Frame.select("habitat").distinct().show()


//*********************************************************************************************************************
                                  /** DATA CLEANSING & PROCESSING STAGE */
//*********************************************************************************************************************

    // Separate label "class" and the features
    // "id", "class" and "veil-type" will be excluded from the dataframe used to train the different models
    // All remaining columns in the dataframe will be used as features for model training

       val label = "class"
       val id = "id"
       val featuresOld1 = for (col <- Frame.columns if ((col != label))) yield col
       val featuresOld2 = for (col <- Frame.columns if ((col != "veil-type"))) yield col
       val features = for (col <- featuresOld1 if ((col != id))) yield col

    // Indexer for label: Create indexes for the label column
       var labelIndexer = new StringIndexer()
        .setInputCol(label)
        .setOutputCol("i_" + label)

    // Indexers for the feature columns: Create indexes for the feature columns
        var featureIndexers = Array[StringIndexer]()
        for (f <- features)
        featureIndexers = featureIndexers :+ new StringIndexer()
          .setInputCol(f)
          .setOutputCol("i_" + f)
          .setHandleInvalid("skip")

    // Generate a feature vector as standard parameter for the Spark-supplied ML algorithms
        val featureColumns = featureIndexers.map(f => f.getOutputCol)

        val assembler = new VectorAssembler()
          .setInputCols(featureColumns)
          .setOutputCol("features")

    // Automatically identify categorical features
    // setMaxCategories: features with more than 12 distinct values are treated as continuous
        val catVectorIndexer = new VectorIndexer()
          .setInputCol(assembler.getOutputCol)
          .setOutputCol("catFeatures")
          .setMaxCategories(12)

//*********************************************************************************************************************
                                       /** TRAIN & TEST DATA SPLIT */
//*********************************************************************************************************************
    // Split of original dataset into a training and test data set
    // 70% training and 30% test data.

         val Array(trainingData, testData) = Frame.randomSplit(Array(0.7, 0.3))

//*********************************************************************************************************************
                                   /** #1 DECISION TREE  CLASSIFIER */
//*********************************************************************************************************************
     // Create a Decision Tree Classifier-Object

      val DecisionTree = new DecisionTreeClassifier()
         .setLabelCol(labelIndexer.getOutputCol)
         .setFeaturesCol(catVectorIndexer.getOutputCol)
         .setPredictionCol("predictedIndex")

//*********************************************************************************************************************
                                 /** #2 RANDOM   FOREST  CLASSIFIER */
//*********************************************************************************************************************
     // Create a Random Forest Classifier-Object

     val rfClassifier = new RandomForestClassifier()
        .setLabelCol(labelIndexer.getOutputCol)
        .setFeaturesCol(catVectorIndexer.getOutputCol)
        .setPredictionCol("predictedIndex")
        .setNumTrees(10)
//*********************************************************************************************************************
                              /** # 3 Gradient-Boosted Tree Classifier */
//*********************************************************************************************************************
      // Create a Gradient-Boosted Tree Classifier-Object

      val gbt = new GBTClassifier()
          .setLabelCol(labelIndexer.getOutputCol)
          .setFeaturesCol(catVectorIndexer.getOutputCol)
          .setMaxIter(10)
          .setFeatureSubsetStrategy("auto")
//*********************************************************************************************************************
                                /** LABEL & PIPELINE & PREDICTIONS */
//*********************************************************************************************************************

    // Mapping of indexes for the label to the original values
      var labels = labelIndexer.fit(Frame).labels

     // Back conversion of values for the predicted label
     var labelConverter = new IndexToString()
       .setInputCol(gbt.getPredictionCol)
       .setOutputCol("predictedLabel")
       .setLabels(labels)

     // Creation of a Machine Learning Pipeline for Mushroom classification
      var pipeline = new Pipeline().setStages(
      Array(labelIndexer) ++
        featureIndexers :+
        assembler :+
        catVectorIndexer :+
        gbt  :+
        labelConverter)

    // Creation of a PipelineModel based on the training data set
      val model = pipeline.fit(trainingData)

    // Application of the created PipelineModel on the test data set
      val predictions = model.transform(Test)
      val predictionsDF = predictions.select("id", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",  "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat", "predictedLabel")

      //predictions.show()
//*******************************************************************************************************************
                                     /** MODEL EVALUATION */
//*******************************************************************************************************************

   // Evaluation of the Pipeline Model

    val originalEvaluatorA = new MulticlassClassificationEvaluator()
      .setLabelCol(labelIndexer.getOutputCol)
      .setPredictionCol(DecisionTree.getPredictionCol)
      .setMetricName("accuracy")

     val originalEvaluatorB = new BinaryClassificationEvaluator()
       .setLabelCol(labelIndexer.getOutputCol)
       .setRawPredictionCol(DecisionTree.getPredictionCol)
       .setMetricName("areaUnderROC")

     val originalEvaluatorC = new MulticlassClassificationEvaluator()
       .setLabelCol(labelIndexer.getOutputCol)
       .setPredictionCol(DecisionTree.getPredictionCol)
       .setMetricName("f1")

     val originalEvaluatorD = new MulticlassClassificationEvaluator()
       .setLabelCol(labelIndexer.getOutputCol)
       .setPredictionCol(DecisionTree.getPredictionCol)
       .setMetricName("weightedRecall")

     val originalEvaluatorE = new MulticlassClassificationEvaluator()
       .setLabelCol(labelIndexer.getOutputCol)
       .setPredictionCol(DecisionTree.getPredictionCol)
       .setMetricName("weightedPrecision")


    val accuracy      = originalEvaluatorA.evaluate(predictions)
    val areaUnderROC  = originalEvaluatorB.evaluate(predictions)
    val F1Score       = originalEvaluatorC.evaluate(predictions)
    val Recall        = originalEvaluatorD.evaluate(predictions)
    val Precision     = originalEvaluatorE.evaluate(predictions)


   // Printing Performance Metrics
    println(f" The Accuracy is : $accuracy%.10f")
    println(f" The Area Under the Curve  is : $areaUnderROC%.10f")
    println(f" The F1 Score  is :  $F1Score%.10f")
    println(f" The Recall  is : $Recall%.10f")
    println(f" The Precision  is :  $Precision%.10f")

    // Write predictions to CSV File
     val PredictionsDF = predictionsDF.select("id", "predictedLabel")
     val PredictionFile = PredictionsDF.withColumnRenamed("predictedLabel", "class")
     PredictionFile.coalesce(1).write.format("com.databricks.spark.csv").mode("overwrite").option("header", "true").save("C:/Users/Dr. Elombe Calvert/IdeaProjects/Hello World/data/ TestDT.csv")

    // Stop the session
    spark.stop()

   }

 }
