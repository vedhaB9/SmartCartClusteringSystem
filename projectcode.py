import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#   1. load the dataset
df=pd.read_csv("smartcart_customers.csv")
pd.set_option('display.max_columns', None)

#print(df.head())




#   2. Read the dataset ,check for any missing values
print('all abt the dataset.....')
'''
print(df.isnull().sum())
print(df.info())
print(df.shape)
print(df.columns)'''



#   3.handling missing values
df['Income']=df['Income'].fillna(df['Income'].median())



#   4.feature engineering

                # age(new feature) - Year_Birth
df['Age']=2026-df['Year_Birth']
                # customer_tenure_days(new feature) :- reference_date(max of Dt_Customer)-Dt_Customer
df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'],dayfirst=True)
reference_date=df['Dt_Customer'].max()
df['Customer_Tenure_Days']=(reference_date-df['Dt_Customer']).dt.days
print(df.columns)
                # Total_Spending(new feature) :- MntWines+ MntFruits + MntMeatProducts +MntFishProducts + MntSweetProducts + MntGoldProds
df['Total_Spending']=df['MntMeatProducts']+df['MntFishProducts']+df['MntSweetProducts']+df['MntWines']+df['MntFruits']+df['MntGoldProds']
                # Total_Children(new feature):- Kidhome+Teenhome
df['Total_Children']=df['Kidhome']+df['Teenhome']
print()

                # Education:- Undergraduate,Graduate,PostGraduate

print(df['Education'].value_counts())
print('Transforming the different classes under Education column into three -  Undergraduate,Graduate,PostGraduate')
df['Education']=df['Education'].replace({
    "Basic":"UnderGraduate",
    "2n Cycle":"UnderGraduate",
    "Graduation":"Graduate",
    "PhD": "PostGraduate",
    "Master":"PostGraduate"
})
print(df['Education'].value_counts())
print()
                # living_with(new feature)  partner,solo - from Marital_Status
print(df['Marital_Status'].value_counts())

df['living_with']=df['Marital_Status'].replace({
    "Married":"partner",
    "Together":"partner",
    "Single":"solo",
    "Divorced":"solo",
     "Widow":"solo",
    "Alone":"solo",
    "Absurd":"solo",
    "YOLO":"solo",
})
print(df['living_with'].value_counts())
print()


#    5. Drop unnecessary columns
print(df.columns)
cols=['ID', 'Year_Birth','Marital_Status', 'Dt_Customer', 'Kidhome']
spending_cols=['MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts','MntGoldProds']
print('Removing all the unnecessary columns......')
df_new=df.drop(columns=cols+spending_cols)

print(df_new.columns)
print(df_new.shape)
print()



#   6.Handling Outliers
cols1=['Income','Recency','Response','Age','Total_Spending']
g=sns.pairplot(df_new[cols1],height=3)
g.fig.set_size_inches(5, 5)
plt.tight_layout()
#plt.show()
print()

print("Dataset with outliers: ",len(df_new))
df_new=df_new[ (df_new['Age']< 90) ]
df_new=df_new[ (df_new['Income']< 600000) ]
print("Dataset without outliers: ",len(df_new))
print()


#   7.feature heatmap-finding correlations,etc
corr=df_new.corr(numeric_only=True)
plt.figure(figsize=(12,15))
sns.heatmap(
    corr,
    annot=True,
    annot_kws={"size": 8},
    cmap="coolwarm"
)
#plt.show()
print()


#       8. feature encoding
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
cat_cols=['Education','living_with']

print('Transforming all the categorial columns via encoding ')
encoded_cols=ohe.fit_transform(df_new[cat_cols])
enc_df=pd.DataFrame(encoded_cols.toarray(),columns=ohe.get_feature_names_out(cat_cols),index=df_new.index)
#print(enc_df.head())
df_encoded=pd.concat([df_new.drop(columns=cat_cols), enc_df],axis=1)
print(df_encoded.shape)

#         9. feature scaling
from sklearn.preprocessing import StandardScaler
X=df_encoded
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)



#          10.visualize
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
X_pca=pca.fit_transform(X_scaled)
print('explained_pca_ratio_: ',pca.explained_variance_ratio_)


fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection="3d")
ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2])
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D Projection')
plt.show()

#  11.Analyze k value -elbow method,silhouette score method
 ### 1.elbow method
from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit_predict(X_pca)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
sns.lineplot(x=range(1,11),y=wcss,marker='o')
plt.xlabel('Hyperparameter k values')
plt.ylabel('wcss')
plt.title('Finding elbow point using elbow method')
plt.show()

from kneed import KneeLocator
knee=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing")
optimal_k=knee.elbow
print(optimal_k)

 ### 2. silhouette score method
from sklearn.metrics import silhouette_score
scores=[]
for k in range(2,11):
    kmeans=KMeans(n_clusters=k)
    labels=kmeans.fit_predict(X_pca)
    score=silhouette_score(X_pca,labels)
    scores.append(score)

plt.figure(figsize=(8,6))
sns.lineplot(x=range(2,11),y=scores,marker='o')
plt.xlabel('Hyperparameter k values')
plt.ylabel('silhouette scores')
plt.title('Finding k value via silhouette score')
plt.show()

    ## combined plot
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(range(2, 11), wcss[1:], marker='o', label='WCSS')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('WCSS')

ax2 = ax1.twinx()

ax2.plot(range(2, 11), scores, marker='o', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score')

plt.title('Elbow and Silhouette Analysis')
plt.show()

#   12. Clustering
            ## KMeans Clustering
kmeans=KMeans(n_clusters=4,random_state=42)
labels_kmeans=kmeans.fit_predict(X_pca)

f1=plt.figure(figsize=(8,6))
ax=f1.add_subplot(111,projection="3d")
ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=labels_kmeans)
ax.set_title('Using KMeans Clustering')


plt.show()
        ## Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
agg=AgglomerativeClustering(n_clusters=4)
labels_agg=agg.fit_predict(X_pca)
f2=plt.figure(figsize=(8,6))
ax=f2.add_subplot(111,projection="3d")
ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2],c=labels_agg)
ax.set_title('Using Agglomerative Clustering')
plt.show()


#13. Characterization of clusters
df_new['cluster']=labels_agg
print(df_new.head())

colors=['green','blue','orange','yellow']
sns.countplot(x=df_new['cluster'],palette=colors,hue=df_new['cluster'])
plt.show()

# 14. cluster summary
cluster_summary=df_new.groupby('cluster').mean(numeric_only=True)
print(cluster_summary)

















