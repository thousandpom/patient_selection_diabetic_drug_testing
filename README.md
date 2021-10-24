# Patient Selection for Diabetic Drug Testing Trial
*This is part of the Udaciy AI for Healthcare Nanodegree course projects.*

### Key Words:
`Electronic_Health_Records`, `TensorFlow_Probability_Layers`, `Sequential_Model`, `Feature Selection`, `AI_Fairness`

The goal of the project is to predict the estimated hospitalization time for patients using their electronic health records (EHR) and to select patients who are likely to stay in hospital for more than five days.     

The dataset used in this project is Udacity adapted synthetic version of the UC Irvine Diabetes dataset. Details about the original dataset can be found at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008). 

The notebook `EHR_main.ipynb` contains the project details and calls the functions to run the EDA of the dataset, preprocess the data, build and train the model, and finally perform a fairness analysis of the final prediction using the Aequitas frameworking. 

