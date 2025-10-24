# Arbitrary Value Imputation in Machine Learning

This project demonstrates how to handle missing data using **Arbitrary Value Imputation** and analyze the effect of **variance shifting** after performing mean and median imputations.


## Topics Covered:-
-> Understanding arbitrary value imputation  
-> Performing mean and median imputations using scikit-learn  
-> Applying arbitrary constants (e.g., 0, -999, or custom values)  
-> Comparing variance and distribution shifts across methods  
-> Observing the impact on model inputs  

## Libraries Used:-
-> pandas  
-> numpy  
-> scikit-learn  
-> matplotlib / seaborn  


## Key Takeaway:-
Arbitrary Value Imputation:
-> Replaces missing values with a fixed constant  
-> Keeps dataset size intact  
-> Can cause **variance shifts** or **distribution distortion**, so it must be used carefully  

By comparing it with **mean** and **median imputation**, we can see how arbitrary constants influence the feature’s spread and scale.


 **Note:**  
-> This notebook uses sample data to demonstrate how arbitrary constants and statistical imputations affect variance, providing insight into when each approach should be used.
