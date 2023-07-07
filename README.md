# Heart Attack Prediction
Heart disease is a general term that means that the heart is not working normally.
This dataset gives a number of variables along with a target condition of having or not having heart disease.

Amongst all the organs, The heart is a significant part of our body.

The heart beats about 2.5 billion times over the average lifetime, pushing millions of gallons of blood to every part of the body.

In this era, heart disease is increasing day by day due to the modern lifestyle and food.

The diagnosis of heart disease is a challenging task.

This classification model will predict whether the patient has heart disease or not based on various conditions/symptoms in their body.

## Data Details
It's a clean, easy to understand set of data. However, the meaning of some of the column headers is not obvious. Here's what they mean,

* **age:** The person's age in years

* **sex:** The person's sex (1 = male, 0 = female)

* **cp:** The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)

* **trestbps:** The person's resting blood pressure (mm Hg on admission to the hospital)

* **chol:** The person's cholesterol measurement in mg/dl

* **fbs:** The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

* **restecg:** Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' 
criteria)

* **thalach:** The person's maximum heart rate achieved

* **exang:** Exercise induced angina (1 = yes; 0 = no)

* **oldpeak:** ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)

* **slope:** the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

* **ca:** The number of major vessels (0-3)

* **thal:** A blood disorder called thalassemia

* **target:** Heart disease (0 = no, 1 = yes)

### Distribution by sex

![image](https://github.com/nepomucenoc/heart_attack_prediction_apache_spark/assets/72771264/b9632292-87ac-4ad7-bd0c-6c625667139a)

### Distribution by by age
![newplot](https://github.com/nepomucenoc/heart_attack_prediction_apache_spark/assets/72771264/c93a7e20-b5b0-4d7f-88f7-3606aa7e20d8)

### Prediction result

Using decision trees for classification, the model had an accuracy of 78%. With the following metrics:

* TP = 43.0
* FP = 15.0
* TN = 35.0
* FN = 7.0
* Precision = 0.7413793
* Recall = 0.86


