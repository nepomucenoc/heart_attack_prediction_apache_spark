// Databricks notebook source
// DBTITLE 1,Business Problem
// MAGIC %md
// MAGIC
// MAGIC ###Heart disease is a general term that means that the heart is not working normally.
// MAGIC
// MAGIC This **dataset** gives a number of variables along with a **target condition** of having or not having heart disease.
// MAGIC
// MAGIC Amongst all the organs, The heart is a significant part of our body. 
// MAGIC
// MAGIC The heart beats about 2.5 billion times over the average lifetime, pushing millions of gallons of blood to every part of the body.
// MAGIC
// MAGIC In this era, the heart disease is increasing day by day due to the modern lifestyle and food. 
// MAGIC
// MAGIC The diagnosis of heart disease is a challenging task. 
// MAGIC
// MAGIC This **classification model will predict whether the patient has heart disease or not** based on various conditions/symptoms of their body.

// COMMAND ----------

// MAGIC %md ### Load Source Data
// MAGIC The data for this project is provided as a CSV file containing details of patient. The data includes specific characteristics (or *features*) for each patient, as well as a column(target) indicating heart attack or not.
// MAGIC
// MAGIC You will load this data into a DataFrame and display it.

// COMMAND ----------

// DBTITLE 1,Code for Loading Data (csv file) to Dataframe 
// MAGIC %scala 
// MAGIC
// MAGIC // File location and type
// MAGIC val file_location = "/FileStore/tables/heart.csv"
// MAGIC val file_type = "csv"
// MAGIC
// MAGIC // CSV options
// MAGIC val infer_schema = "true"
// MAGIC val first_row_is_header = "true"
// MAGIC val delimiter = ","
// MAGIC
// MAGIC // The applied options are for CSV files. For other file types, these will be ignored.
// MAGIC val heartDF = spark.read.format(file_type) 
// MAGIC   .option("inferSchema", infer_schema) 
// MAGIC   .option("header", first_row_is_header) 
// MAGIC   .option("sep", delimiter) 
// MAGIC   .load(file_location)
// MAGIC
// MAGIC heartDF.show()

// COMMAND ----------

// DBTITLE 1,Data Level Details
// MAGIC %md
// MAGIC
// MAGIC It's a clean, easy to understand set of data. However, the meaning of some of the column headers are not obvious. Here's what they mean,
// MAGIC
// MAGIC * **age:** The person's age in years
// MAGIC
// MAGIC * **sex:** The person's sex (1 = male, 0 = female)
// MAGIC
// MAGIC * **cp:** The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
// MAGIC
// MAGIC * **trestbps:** The person's resting blood pressure (mm Hg on admission to the hospital)
// MAGIC
// MAGIC * **chol:** The person's cholesterol measurement in mg/dl
// MAGIC
// MAGIC * **fbs:** The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
// MAGIC
// MAGIC * **restecg:** Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' 
// MAGIC criteria)
// MAGIC
// MAGIC * **thalach:** The person's maximum heart rate achieved
// MAGIC
// MAGIC * **exang:** Exercise induced angina (1 = yes; 0 = no)
// MAGIC
// MAGIC * **oldpeak:** ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
// MAGIC
// MAGIC * **slope:** the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
// MAGIC
// MAGIC * **ca:** The number of major vessels (0-3)
// MAGIC
// MAGIC * **thal:** A blood disorder called thalassemia
// MAGIC
// MAGIC * **target:** Heart disease (0 = no, 1 = yes)

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC **Diagnosis:** The diagnosis of heart disease is done on a combination of clinical signs and test results. The types of tests run will be chosen on the basis of what the physician thinks is going on 1, ranging from electrocardiograms and cardiac computerized tomography (CT) scans, to blood tests and exercise stress tests 2.
// MAGIC
// MAGIC Looking at information of heart disease risk factors led me to the following: high cholesterol, high blood pressure, diabetes, weight, family history and smoking 3. According to another source 4, the major factors that can't be changed are: increasing age, male gender and heredity. Note that thalassemia, one of the variables in this dataset, is heredity. Major factors that can be modified are: Smoking, high cholesterol, high blood pressure, physical inactivity, and being overweight and having diabetes. Other factors include stress, alcohol and poor diet/nutrition.
// MAGIC
// MAGIC I can see no reference to the 'number of major vessels', but given that the definition of heart disease is "...what happens when your heart's blood supply is blocked or interrupted by a build-up of fatty substances in the coronary arteries", it seems logical the more major vessels is a good thing, and therefore will reduce the probability of heart disease.
// MAGIC
// MAGIC Given the above, I would hypothesis that, if the model has some predictive ability, we'll see these factors standing out as the most important.

// COMMAND ----------

// DBTITLE 1,Record Count
// MAGIC %scala
// MAGIC
// MAGIC heartDF.count()

// COMMAND ----------

// DBTITLE 1,Finding count, mean, maximum, standard deviation and minimum
// MAGIC %scala
// MAGIC
// MAGIC heartDF.select("age", "sex", "cp", "trestbps", "chol", "fbs","restecg", "thalach","exang","oldpeak","slope","ca","thal","target").describe().show()
// MAGIC

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC heartDF.describe().show()

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC heartDF.select("age", "sex", "cp", "trestbps").describe().show()

// COMMAND ----------

// DBTITLE 1,Printing Schema
// MAGIC %scala
// MAGIC
// MAGIC heartDF.printSchema()

// COMMAND ----------

// DBTITLE 1,Creating Temp View from Dataframe 
// MAGIC %scala
// MAGIC
// MAGIC heartDF.createOrReplaceTempView("HeartData");

// COMMAND ----------

// DBTITLE 1,Querying the Temporary View
// MAGIC %sql
// MAGIC
// MAGIC select * from HeartData;

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC #Exploratory Data Analysis

// COMMAND ----------

// DBTITLE 1,Sex Result
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(sex),
// MAGIC CASE
// MAGIC     WHEN sex == 1 THEN "Male"
// MAGIC     ELSE "Female"
// MAGIC END AS sex
// MAGIC FROM HeartData group by sex;

// COMMAND ----------

// DBTITLE 1,Chest Pain Type
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(cp),
// MAGIC CASE
// MAGIC     WHEN cp == 0 THEN "Typical angina"
// MAGIC     WHEN cp == 1 THEN "Atypical angina"
// MAGIC     WHEN cp == 2 THEN "Non-anginal pain"
// MAGIC     ELSE "Asymptomatic"
// MAGIC END AS ChestPain
// MAGIC FROM HeartData group by cp;

// COMMAND ----------

// DBTITLE 1,Fasting Blood Sugar
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(fbs),
// MAGIC CASE
// MAGIC     WHEN fbs == 1 THEN "True"
// MAGIC     ELSE "False"
// MAGIC END AS FastingBloodSugar
// MAGIC FROM HeartData group by fbs;

// COMMAND ----------

// DBTITLE 1,Resting Electro Cardio Graphic
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(restecg),
// MAGIC CASE
// MAGIC     WHEN restecg == 0 THEN "Normal"
// MAGIC     WHEN restecg == 1 THEN "Abnormality"
// MAGIC     ELSE "Left Ventricular Hypertrophy"
// MAGIC END AS RestingElectrocardioGraphic
// MAGIC FROM HeartData group by restecg;

// COMMAND ----------

// DBTITLE 1,Exercise Induced Angina
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(exang),
// MAGIC CASE
// MAGIC     WHEN exang == 0 THEN "Not Induced"
// MAGIC     ELSE "Induced"
// MAGIC END AS ExerciseInducedAngina
// MAGIC FROM HeartData group by exang;
// MAGIC

// COMMAND ----------

// DBTITLE 1,Slope Result
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(slope),
// MAGIC CASE
// MAGIC     WHEN slope == 0 THEN "Upsloping"
// MAGIC     WHEN slope == 1 THEN "Flat"
// MAGIC     ELSE "Downsloping"
// MAGIC END AS Slope
// MAGIC FROM HeartData group by slope;

// COMMAND ----------

// DBTITLE 1,The number of major vessels (0-4)
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(ca), ca FROM HeartData group by ca order by ca;

// COMMAND ----------

// DBTITLE 1,Blood Disorder called Thalassemia
// MAGIC %sql
// MAGIC
// MAGIC select thal, count(thal) FROM HeartData group by thal order by thal;

// COMMAND ----------

// DBTITLE 1,Disease Status vs Sex
// MAGIC %sql
// MAGIC
// MAGIC select count(sex),
// MAGIC CASE
// MAGIC     WHEN sex == 0 THEN "Female"
// MAGIC     ELSE "Male"
// MAGIC END AS Sex,
// MAGIC count(target), 
// MAGIC CASE
// MAGIC     WHEN target == 0 THEN "No Disease"
// MAGIC     ELSE "Has Disease"
// MAGIC   END AS DiseaseStatus
// MAGIC from HeartData group by sex, target;

// COMMAND ----------

// DBTITLE 1,Disease Status vs Chest Pain Type 
// MAGIC %sql
// MAGIC
// MAGIC select count(cp),
// MAGIC CASE
// MAGIC     WHEN cp == 0 THEN "Typical angina"
// MAGIC     WHEN cp == 1 THEN "Atypical angina"
// MAGIC     WHEN cp == 2 THEN "Non-anginal pain"
// MAGIC     ELSE "Asymptomatic"
// MAGIC END AS ChestPain,
// MAGIC count(target), 
// MAGIC CASE
// MAGIC     WHEN target == 0 THEN "No Disease"
// MAGIC     ELSE "Has Disease"
// MAGIC   END AS DiseaseStatus
// MAGIC from HeartData group by cp, target order by cp;

// COMMAND ----------

// DBTITLE 1,Disease Status vs Fasting Blood Sugar
// MAGIC %sql
// MAGIC
// MAGIC select count(fbs),
// MAGIC CASE
// MAGIC     WHEN fbs == 1 THEN "True"
// MAGIC     ELSE "False"
// MAGIC END AS FastingBloodSugar,
// MAGIC count(target), 
// MAGIC CASE
// MAGIC     WHEN target == 0 THEN "No Disease"
// MAGIC     ELSE "Has Disease"
// MAGIC   END AS DiseaseStatus
// MAGIC from HeartData group by fbs, target order by fbs;

// COMMAND ----------

// DBTITLE 1,Disease Status vs Resting Electro Cardio Graphic Results
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(restecg),
// MAGIC CASE
// MAGIC     WHEN restecg == 0 THEN "Normal"
// MAGIC     WHEN restecg == 1 THEN "Abnormality"
// MAGIC     ELSE "Left Ventricular Hypertrophy"
// MAGIC END AS RestingElectrocardioGraphic,
// MAGIC count(target), 
// MAGIC CASE
// MAGIC     WHEN target == 0 THEN "No Disease"
// MAGIC     ELSE "Has Disease"
// MAGIC   END AS DiseaseStatus
// MAGIC from HeartData group by restecg, target order by restecg;

// COMMAND ----------

// DBTITLE 1,Disease Status vs Exercise Induced Angina
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(exang),
// MAGIC CASE
// MAGIC     WHEN exang == 0 THEN "Not Induced"
// MAGIC     ELSE "Induced"
// MAGIC END AS ExerciseInducedAngina,
// MAGIC count(target), 
// MAGIC CASE
// MAGIC     WHEN target == 0 THEN "No Disease"
// MAGIC     ELSE "Has Disease"
// MAGIC   END AS DiseaseStatus
// MAGIC from HeartData group by exang, target order by exang;

// COMMAND ----------

// DBTITLE 1,Disease Status vs Slope
// MAGIC %sql
// MAGIC
// MAGIC SELECT count(slope),
// MAGIC CASE
// MAGIC     WHEN slope == 0 THEN "Upsloping"
// MAGIC     WHEN slope == 1 THEN "Flat"
// MAGIC     ELSE "Downsloping"
// MAGIC END AS Slope,
// MAGIC count(target), 
// MAGIC CASE
// MAGIC     WHEN target == 0 THEN "No Disease"
// MAGIC     ELSE "Has Disease"
// MAGIC   END AS DiseaseStatus
// MAGIC from HeartData group by slope, target order by slope;

// COMMAND ----------

// DBTITLE 1,Disease Status vs CA
// MAGIC %sql
// MAGIC
// MAGIC SELECT ca, count(ca),
// MAGIC count(target), 
// MAGIC CASE
// MAGIC     WHEN target == 0 THEN "No Disease"
// MAGIC     ELSE "Has Disease"
// MAGIC   END AS DiseaseStatus
// MAGIC from HeartData group by ca, target order by ca;

// COMMAND ----------

// DBTITLE 1,Disease Status vs Thal
// MAGIC %sql
// MAGIC
// MAGIC SELECT thal, count(thal),count(target), 
// MAGIC CASE
// MAGIC     WHEN target == 0 THEN "No Disease"
// MAGIC     ELSE "Has Disease"
// MAGIC   END AS DiseaseStatus
// MAGIC from HeartData group by thal, target order by thal;

// COMMAND ----------

// DBTITLE 1,Histogram of Age
// MAGIC %sql
// MAGIC
// MAGIC select age from HeartData;

// COMMAND ----------

// DBTITLE 1,Histogram of Blood Pressure
// MAGIC %sql
// MAGIC
// MAGIC select trestbps from HeartData;

// COMMAND ----------

// DBTITLE 1,Histogram of Cholesterol
// MAGIC %sql
// MAGIC
// MAGIC select chol from HeartData;

// COMMAND ----------

// DBTITLE 1,Histogram of Maximum Heart Rate
// MAGIC %sql
// MAGIC
// MAGIC select thalach from HeartData;

// COMMAND ----------

// DBTITLE 1,Histogram of ST Depression
// MAGIC %sql
// MAGIC
// MAGIC select oldpeak from HeartData;

// COMMAND ----------

// DBTITLE 1,Histogram of Heart disease
// MAGIC %sql
// MAGIC
// MAGIC select target from HeartData;

// COMMAND ----------

// DBTITLE 1,Age Result
// MAGIC %sql
// MAGIC
// MAGIC select age,count(age) as AgeCounter from HeartData group by age order by age;

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC SELECT age, target, COUNT(*) AS AgeCounter
// MAGIC FROM HeartData
// MAGIC GROUP BY age, target
// MAGIC ORDER BY age;
// MAGIC

// COMMAND ----------

// DBTITLE 1,Age Distribution Result
// MAGIC %sql
// MAGIC
// MAGIC SELECT 
// MAGIC CASE
// MAGIC     WHEN age >=29 and age <40 THEN "Young Ages"
// MAGIC     WHEN age >=40 and age <55 THEN "Middle Ages"
// MAGIC     ELSE "Elderly Ages"
// MAGIC   END AS Age,
// MAGIC   count(Age)
// MAGIC from HeartData group by Age;

// COMMAND ----------

// DBTITLE 1,Age Distribution with respect to Sex
// MAGIC %sql
// MAGIC
// MAGIC SELECT 
// MAGIC CASE
// MAGIC     WHEN age >=29 and age <40 THEN "Young Ages"
// MAGIC     WHEN age >=40 and age <55 THEN "Middle Ages"
// MAGIC     ELSE "Elderly Ages"
// MAGIC   END AS Age,
// MAGIC CASE
// MAGIC     WHEN sex == 1 THEN "Male"
// MAGIC     ELSE "Female"
// MAGIC END AS sex
// MAGIC from HeartData;

// COMMAND ----------

// DBTITLE 1,Age VS Blood Pressure
// MAGIC %sql
// MAGIC
// MAGIC select trestbps, age from HeartData;

// COMMAND ----------

// DBTITLE 1,One Visualization to Rule Them All
// MAGIC %sql
// MAGIC
// MAGIC select age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target from HeartData;

// COMMAND ----------

// DBTITLE 1,Age VS Maximum Heart Rate
// MAGIC %sql
// MAGIC
// MAGIC select age, thalach from HeartData order by age asc; 

// COMMAND ----------

// DBTITLE 1,Age VS Heart Diseases
// MAGIC %sql
// MAGIC
// MAGIC select target, age, count(age), count(target) from HeartData group by age, target;

// COMMAND ----------

// MAGIC %md ## Creating a Classification Model
// MAGIC
// MAGIC In this Project, you will implement a classification model **(Decision tree classifier)** that uses features of patient details and we will predict it is heart diseases (Yes or No)
// MAGIC
// MAGIC ### Import Spark SQL and Spark ML Libraries
// MAGIC
// MAGIC First, import the libraries you will need:

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC import org.apache.spark.sql.types._
// MAGIC import org.apache.spark.sql.functions._
// MAGIC
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassificationModel
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassifier
// MAGIC import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// MAGIC import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

// MAGIC %md ### Prepare the Training Data
// MAGIC To train the classification model, you need a training data set that includes a vector of numeric features, and a label column. In this project, you will use the **VectorAssembler** class to transform the feature columns into a vector, and then rename the **Attrition** column to **label**.

// COMMAND ----------

// MAGIC %md ###VectorAssembler()
// MAGIC
// MAGIC VectorAssembler():  is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. 
// MAGIC
// MAGIC **VectorAssembler** accepts the following input column types: **all numeric types, boolean type, and vector type.** 
// MAGIC
// MAGIC In each row, the **values of the input columns will be concatenated into a vector** in the specified order.

// COMMAND ----------

// MAGIC %md ### Split the Data
// MAGIC It is common practice when building machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this project, you will use 70% of the data for training, and reserve 30% for testing. 

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val splits = heartDF.randomSplit(Array(0.7, 0.3))
// MAGIC val train = splits(0)
// MAGIC val test = splits(1)
// MAGIC val train_rows = train.count()
// MAGIC val test_rows = test.count()
// MAGIC println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC import org.apache.spark.ml.feature.VectorAssembler
// MAGIC
// MAGIC val assembler = new VectorAssembler().setInputCols(Array("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal")).setOutputCol("features")
// MAGIC
// MAGIC val training = assembler.transform(train).select($"features", $"target".alias("label"))
// MAGIC
// MAGIC training.show()

// COMMAND ----------

// MAGIC %md ### Train a Classification Model (Decision tree classifier)
// MAGIC Next, you need to train a Classification Model using the training data. To do this, create an instance of the Decision tree classifier algorithm you want to use and use its **fit** method to train a model based on the training DataFrame. In this Project, you will use a *Decision tree classifier* algorithm 

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassificationModel
// MAGIC import org.apache.spark.ml.classification.DecisionTreeClassifier
// MAGIC import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// MAGIC
// MAGIC val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
// MAGIC
// MAGIC val model = dt.fit(training)
// MAGIC
// MAGIC println("Model Trained!")

// COMMAND ----------

// MAGIC %md ### Prepare the Testing Data
// MAGIC Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the **target** column to **trueLabel**.

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val testing = assembler.transform(test).select($"features", $"target".alias("trueLabel"))
// MAGIC testing.show()

// COMMAND ----------

// MAGIC %md ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. But in this case you are using the test data which includes a known true label value, so you can compare the predicted target. 

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val prediction = model.transform(testing)
// MAGIC val predicted = prediction.select("features", "prediction", "trueLabel")
// MAGIC predicted.show(100)

// COMMAND ----------

// MAGIC %md Looking at the result, the **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data. It looks like there is some variance between the predictions and the actual values (the individual differences are referred to as *residuals*) you'll learn how to measure the accuracy of a model.

// COMMAND ----------

// MAGIC %md ### Classification model Evalation
// MAGIC
// MAGIC spark.mllib comes with a number of machine learning algorithms that can be used to learn from and make predictions on data. When these algorithms are applied to build machine learning models, there is a need to evaluate the performance of the model on some criteria, which depends on the application and its requirements. spark.mllib also provides a suite of metrics for the purpose of evaluating the performance of machine learning models.

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val evaluator = new MulticlassClassificationEvaluator()
// MAGIC   .setLabelCol("trueLabel")
// MAGIC   .setPredictionCol("prediction")
// MAGIC   .setMetricName("accuracy")
// MAGIC val accuracy = evaluator.evaluate(prediction)
// MAGIC
// MAGIC

// COMMAND ----------

// MAGIC %md ### Compute Confusion Matrix Metrics
// MAGIC Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
// MAGIC - True Positives
// MAGIC - True Negatives
// MAGIC - False Positives
// MAGIC - False Negatives
// MAGIC
// MAGIC From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val tp = predicted.filter("prediction == 1 AND truelabel == 1").count().toFloat
// MAGIC val fp = predicted.filter("prediction == 1 AND truelabel == 0").count().toFloat
// MAGIC val tn = predicted.filter("prediction == 0 AND truelabel == 0").count().toFloat
// MAGIC val fn = predicted.filter("prediction == 0 AND truelabel == 1").count().toFloat
// MAGIC val metrics = spark.createDataFrame(Seq(
// MAGIC  ("TP", tp),
// MAGIC  ("FP", fp),
// MAGIC  ("TN", tn),
// MAGIC  ("FN", fn),
// MAGIC  ("Precision", tp / (tp + fp)),
// MAGIC  ("Recall", tp / (tp + fn)))).toDF("metric", "value")
// MAGIC metrics.show()
