# CSE 572: Project 2 - Meal / No Meal Classification (Group 30)

## Requirements/Dependencies
python3, tensorflow, keras, sklearn, numpy, pandas, matplotlib, impyute, datawig, tsfresh, scipy

## Execution Instructions
* Start Execution

  ```python main.py```

* **For Test**
  * In the terminal/command line paste the **absolute or realtive path** of the test file.
  * Test file should be in **CSV** format (structure similar to data provided)
  * Example
    ``` Input filename (Enter 0 to exit): MealNoMealData/mealData1.csv```
  * Output:
    * K Fold results (for each model)
      * Accuracy
      * Precision
      * Recall
      * F1 Score
    * Prediction on new dataset
      * **1** - Meal
      * **0** - No Meal


## Description
When the program is executed 4 models are tranied and cross validated.

### Execution Flow
  * Data Cleaning
  * Feature extraction & PCA
  * K-fold cross validation
    * Sumultaneous training of the models
    * Testing of the models
  * Training on given dataset
  * Testing on custom input

### Models:
  * SVM
  * ANN
  * Ada Boost
  * Decision Tree
  
### Value of K in K-Fold
  * 4
