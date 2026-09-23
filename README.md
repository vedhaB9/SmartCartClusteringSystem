# SmartCart Customer Clustering System

## 📌 Project Overview

SmartCart Customer Clustering System is a machine learning project that segments customers into different groups based on their demographic characteristics, purchasing behavior, and customer-related attributes.

The main objective is to identify meaningful customer segments that can help businesses better understand their customers and support data-driven marketing strategies.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- KMeans Clustering
- Agglomerative Clustering
- Principal Component Analysis (PCA)

## 🔄 Project Workflow

1. Data Collection
2. Data Preprocessing
3. Handling Missing Values
4. Feature Engineering
5. Outlier Detection and Removal
6. Categorical Feature Encoding
7. Feature Scaling
8. Dimensionality Reduction using PCA
9. Finding the Optimal Number of Clusters
10. Customer Clustering
11. Cluster Analysis

## 🧹 Data Preprocessing

The dataset was cleaned and prepared for clustering by:

- Handling missing values
- Removing unnecessary columns
- Converting customer registration dates
- Creating customer age
- Calculating customer tenure
- Calculating total spending
- Calculating total number of children
- Grouping education categories
- Grouping marital status into living-with categories
- Handling outliers

## ⚙️ Feature Engineering

New features were created to better represent customer behavior:

- **Age** – Customer age
- **Customer_Tenure_Days** – Number of days since customer registration
- **Total_Spending** – Total amount spent across product categories
- **Total_Children** – Total number of children in the household

## 📊 Dimensionality Reduction

Principal Component Analysis (PCA) was applied after feature scaling to reduce the dimensionality of the dataset and visualize customer clusters in a lower-dimensional space.

## 🔍 Finding the Optimal Number of Clusters

Two techniques were used to evaluate the suitable number of clusters:

- Elbow Method
- Silhouette Score

These methods were used to analyze clustering performance and determine a suitable value of K.

## 🤖 Clustering Algorithms

### K-Means Clustering

K-Means was used to divide customers into groups based on similarity in their feature values.

### Agglomerative Clustering

Agglomerative Hierarchical Clustering was also applied to compare customer segmentation using a different clustering approach.

## 📈 Cluster Analysis

After clustering, the characteristics of each customer group were analyzed using numerical summaries to understand the differences between the identified customer segments.

## 🎯 Project Objective

The objective of this project is to identify customer segments based on their characteristics and purchasing behavior, which can help businesses understand different customer groups and develop more targeted strategies.

## 🚀 Future Improvements

- Experiment with additional clustering algorithms
- Improve feature selection
- Perform deeper cluster profiling
- Build interactive visualizations
- Develop a customer segmentation dashboard
- Evaluate clustering stability on new customer data

## 👩‍💻 Author

**Vedha Shree**

B.Tech – Computer Science and Engineering