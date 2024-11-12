# Ch.2: Machine Learning and Deep Learning 

---

### 1. Case 1: Customer Exit Prediction

This project is dedicated to developing a machine learning classification model aimed at predicting whether a bank customer is likely to leave the bank (churn) based on various features provided in a banking dataset. The dataset contains customer-specific attributes and data points that influence their decision to stay or exit. The goal is to create an accurate predictive system that financial institutions can use to understand customer behavior and proactively address potential churn.

- **Dataset**: [SC_HW1_bank_data.csv](https://raw.githubusercontent.com/Rietaros/kampus_merdeka/main/SC_HW1_bank_data.csv)
  
- **Libraries**: `Pandas`, `NumPy`, `Scikit-learn`
- **Objective**: Classification with Supervised Machine Learning Models

- **Key Objectives**: 
  - *Data Preprocessing*: Clean and prepare the dataset for modeling. This includes handling missing data, encoding categorical variables, and scaling numerical features.
  - *Feature Selection*: Identify which features contribute the most to predicting customer churn. This helps in building a more efficient and accurate model.
  - *Model Development*: Train and evaluate various classification algorithms, such as logistic regression, decision trees, random forests, and gradient boosting, to determine the best-performing model for the task.
  - *Performance Metrics*: Utilize metrics like accuracy, precision, recall, F1-score, and AUC-ROC to measure the model's performance.
  - *Interpretability*: Implement tools such as feature importance plots or SHAP (SHapley Additive exPlanations) values to interpret the model and understand the main drivers behind customer churn.

---

### 2. Case 2: Data Segmentation

This project focuses on the use of unsupervised learning to perform customer segmentation using the KMeans clustering algorithm. The aim is to divide customer data into meaningful groups or clusters based on similarities in their features. By grouping customers with similar characteristics, businesses can tailor marketing strategies, identify customer preferences, and improve service offerings.

- **Dataset**: [cluster_s1.csv](https://raw.githubusercontent.com/Rietaros/kampus_merdeka/main/cluster_s1.csv)
  
- **Libraries**: `Pandas`, `Numpy`, `Scikit-learn`, `Matplotlib`, `Seaborn`

- **Key Objectives**: 
  -*Data Preparation*: Ensure the dataset is clean and ready for clustering, which involves handling missing values, standardizing data, and transforming features as needed.
  - *Clustering with KMeans*: Apply the KMeans algorithm to identify distinct customer groups based on various features in the dataset.
  - *Cluster Evaluation*: Use appropriate metrics and visualization techniques to evaluate the quality of clusters and interpret the results.
  - *Business Insights*: Analyze the clusters to derive actionable insights that can inform business decisions, such as personalized marketing campaigns or customer retention efforts.

---

### 3. Case 3: California House Price Prediction with Neural Networks

This project involves using a neural network model constructed with TensorFlow-Keras to predict house prices based on the California housing dataset. The goal is to develop a predictive model that can accurately estimate the price of a house based on various influential features provided in the dataset.

- **Task**: REGRESSION
- **DL Framework**: Tensorflow-Keras
- **Dataset**: California House Price dataset from `Scikit-Learn`
- **Libraries**: `Pandas`, `Numpy`, `Scikit-learn`, `Matplotlib`
- **Objective** : Predict House Pricing with Dual Input Settings using Multilayer Perceptron

- **Key Objectives**: 
  - *Data Preparation*: Ensure the dataset is prepared for use by handling any missing or irrelevant data, scaling features, and transforming data where necessary.
  - *Model Construction*: Build and configure a neural network using TensorFlow-Keras, selecting the appropriate architecture for regression tasks.
  - *Model Training and Optimization*: Train the model on the dataset and optimize it using techniques like early stopping, tuning the learning rate, and selecting the right activation functions.
  - *Evaluation and Prediction*: Assess the model’s performance using appropriate metrics for regression and make predictions on new data.
  - *Insights and Analysis*: Understand the model’s behavior and feature influence on house price predictions.

---

### 4. Case 4: Fraud Detection in Credit Card Transactions

This project focuses on developing a classification model using PyTorch to identify fraudulent transactions in a credit card dataset. The goal is to construct a robust predictive model that can effectively distinguish between legitimate and fraudulent transactions based on various features extracted from the dataset.

- **Task**: CLASSIFICATION
- **DL Framework**: PyTorch
- **Dataset**: Credit Card Fraud 2023
- **Libraries**: `Pandas`/`cuDF`, `Scikit-learn`/`cuML`, `Numpy`/`cuPy`
- **Objective** : Predict House Pricing with Dual Input Settings using Multilayer Perceptron

- **Key Objectives**: 
  - *Data Preprocessing*: Clean and prepare the dataset for training by handling imbalances, scaling features, and transforming the data appropriately.
  - *Model Development*: Create and fine-tune a classification model using PyTorch, selecting an appropriate architecture for effective fraud detection.
  - *Training and Optimization*: Train the model while applying regularization and optimization techniques to improve performance and reduce overfitting.
  - *Evaluation*: Assess the model’s accuracy and effectiveness using metrics that consider class imbalances.
  - *Insights and Application*: Extract and interpret the model’s findings to provide actionable insights for stakeholders involved in fraud prevention.
---

## Running the Notebooks

1. Open [Google Colab](https://colab.research.google.com/) and upload the `.ipynb` files.
2. Or you can click on the "`Open in Colab`" button at the top of the project on GitHub.


---
