<h2><center><font color="blue"><b>House Saleprice Predictions</b></font></center></h2>
<hr>

### **Domain**

House prices data set assembled and published by Dean De Cock (http://jse.amstat.org/v19n3/decock.pdf). 

### **Data Description**

A set of 2,930 observations with 82 attributes each. 

### **Problem Statement**

The objective of this data set is to use the first 2,430 observations to fit and evaluate different models and use them to make predictions for the last 500 ones. In addition three different models ranging in complexity are evaluated:

* A simple model with two variables (three with the target variable)
* An intermediate model (between 10 and 20 variables)
* A complex model with all variables

### **Solution Statement**

This dataset can be solved by applying regression techniques. Here OLS, Ridge and Hubber models have been tested in four different train-validation splittings. 

### **Benchmark Model**

Ridge regressor on complex model. 

### **Performance Metric**
R<sup>2</sup> coefficient, Mean Absolute Error (MAE)
