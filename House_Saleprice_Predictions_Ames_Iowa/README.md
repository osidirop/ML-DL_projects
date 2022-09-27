

<h1><center><font color="blue"><b>House Saleprice Predictions - Ames Iowa dataset</b></font></center></h1>
<hr>


## **The notebook is structured as following:** 

---
## PART 1.
---

### **A. General overview of the dataset**

The whole dataset under study is overviewed and the features are separated in 4 categories according to the documentation: 
1. continuous
2. discrete
3. nominal 
4. ordinal 

in order to be examined independently and in correlation with other features when is needed. 

### **B. Continuous Features**  
### **C. Discrete Features**
### **D. Nominal Features** 
### **E. Ordinal Features** 


In all the above four sections:
1. EDA 
2. Data cleaning
3. Feature encoding (mainly for ordinal and nominal)
4. Feature engineering 
        
is performed separately for each category of features. Each section contains an Experimentation subsection where different ideas (steps) are applied and the performance of a Ridge model is checked after each step. At these sections, the dataset is split into training and test sets (50-50) and the model is fitted with the default regularization parameter. Each idea-step is implemented with a custom transformer that works with dataframes and the same transformers are also used later in sections G, H and I. 

---
## PART 2
---

### **F. Define functions to extract results from models**

This section contains several functions that are needed to order to extract results from the models. 
    
### **G. Complex model with 249 features**
### **H. Intermediate Model with 50 features**
### **I. Simple model with 2 features**    
    
In the above three sections the following linear models are tested:
    1. Linear Regression (mainly used as a reference; input matrix is fully ranked only in the simple model) 
    2. Ridge with Grid Search
    3. Lasso with Grid Search
    4. Huber (only for the simple model)

For each model, the dataset is splitted in 80-20, 70-30, 60-40, 50-50 (train and validation sets accordingly) with 10 different random seeds. For the complex and intermediate models, the number of iterations for Lasso had to be increased by one order of magnitude in order for the models to converge. The warm_start flag was also switched on. Each section closes with a comparison among the different models. For the regression model that gives the best results, the predicted values for the test set are calculated taking the mean predicted value across the different splittings (same as weighted mean). 


### **J. Final Comparison** 
The final results among the complex, intermediate and simple model are presented in this section. 
    
***
