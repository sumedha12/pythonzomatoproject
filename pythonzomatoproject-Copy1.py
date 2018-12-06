
# coding: utf-8

# <h1><center>PYTHON ZOMATO PROJECT</h1> 
# 

# In[16]:

# Import all the packages and read .csv file from desktop
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/vineet.khattar/Downloads/zomato.csv/zomato.csv',encoding = "ISO-8859-1")
df.head()


# # <font size =4>Create dummy variables from the data</font> 
# 

# In[17]:

dummy1 = pd.get_dummies(df['Has Online delivery'])
df = pd.concat([df,dummy1],axis =1)
df.rename(columns={'No':'noonlinedelivery'}, inplace=True)
df.rename(columns={'Yes':'yesonlinedelivery'}, inplace=True)


# In[18]:

dummy2 = pd.get_dummies(df['Has Table booking'])
df = pd.concat([df,dummy2],axis =1)
df.rename(columns={'No':'Hastable'}, inplace=True)
df.rename(columns={'Yes':'notable'}, inplace=True)


# In[19]:

dummy3 = pd.get_dummies(df['Is delivering now'])
df = pd.concat([df,dummy3],axis =1)
df.rename(columns={'No':'yesdelivery'}, inplace=True)
df.rename(columns={'Yes':'nodelivery'}, inplace=True)


# In[20]:

dummy5 = pd.get_dummies(df['Country Code'])
df = pd.concat([df,dummy5],axis =1)
df.rename(columns={162:'Botswana'}, inplace=True)
df.rename(columns={30:'Brazil'}, inplace=True)
df.rename(columns={216:'United States'}, inplace=True)
df.rename(columns={14:'Australia'}, inplace=True)
df.rename(columns={37:'Canada'}, inplace=True)
df.rename(columns={215:'United Kingdom'}, inplace=True)
df.rename(columns={214:'UAE'}, inplace=True)
df.rename(columns={208:'Turkey'}, inplace=True)
df.rename(columns={191:'Sri Lanka'}, inplace=True)
df.rename(columns={184:'Singapore'}, inplace=True)
df.rename(columns={189:'South Africa'}, inplace=True)
df.rename(columns={166:'Qatar'}, inplace=True)
df.rename(columns={162:'Philippines'}, inplace=True)
df.rename(columns={14:'India'}, inplace=True)
df.rename(columns={1:'India'}, inplace=True)
df.rename(columns={94:'Indonesia'}, inplace=True)
df.rename(columns={148:'148'}, inplace=True)


# In[21]:

dummy6 = pd.get_dummies(df['Has Table booking'])
df = pd.concat([df,dummy6],axis =1)
df.rename(columns={'No':'yesorder'}, inplace=True)
df.rename(columns={'Yes':'noorder'}, inplace=True)


# # <font size =4>Before creating models, I did some EDA on the data<font>

# In[50]:

df.describe()


# <h3 align ='center'>Histogram of Aggregated rating </h3>

# In[44]:

num_bins = 10
plt.hist(df['Aggregate rating'], num_bins, normed=1, facecolor='blue', alpha=0.5)
plt.xlabel("Rating")
plt.ylabel("Counts")
plt.show()


# <font>From the histogram we can conclude that majority ofthe rating is spread out between 2 to 5</font>

# <h3 align ='center'>Histogram of Number of votes</h3>

# In[46]:

num_bins = 10
plt.hist(df['Votes'], num_bins, normed=1, facecolor='blue', alpha=0.5)
plt.xlabel("Votes")
plt.ylabel("Votes")

plt.show()


# From the above model we can say that majority of votes are betwenn 0-1000 and few between 1000-2000

# <h3 align ='center'>SCATTERPLOT</h3>

# In[53]:

x = df['Votes']
y =df['Aggregate rating']
colors = (0,0,0)
 
# Plot
plt.scatter(x, y,  c=colors, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('Votes')
plt.ylabel('Aggregate rating')
plt.show()


# The above scatter diagram shows the relation between number of votes and aggregating rating. We can see that majority of votes are between 0-2000. Additionally, rating is between 2-5

# <h3 align ='center'>LINEAR REGRESSION  </h3>

# In[54]:

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

X = df[['Average Cost for two','yesonlinedelivery','Votes','yesdelivery','Hastable','yesorder','Price range', 'India','Australia','Brazil','Canada','Indonesia','Botswana','Qatar','Singapore','South Africa','Sri Lanka','Turkey','UAE','United Kingdom','United States']]
y = df[['Aggregate rating']]
X = scale(X)
y = scale(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=31)
reg_all = LinearRegression()

reg_all.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)
print(metrics.mean_absolute_error(y_test, y_pred))
print(metrics.mean_squared_error(y_test, y_pred))
print("R^2: {}".format(reg_all.score(X_test, y_test)))



# <font size =3>We get an accuracy of 32%. Mean square error is 0.68. Error goes down after scaling is done on the data</font>

# <h2 align ='center'>KNN CLASSIFICATION</h2>
# 
# To get a better accuracy, I perform KNN classification model. It classifies data as per the rating labels (such as Excellent, Average, Good, Very good )

# In[12]:

from sklearn.neighbors import KNeighborsClassifier
from sk
learn.metrics import classification_report

# Initialize call to ML Algorithm of choice
knn = KNeighborsClassifier(n_neighbors = 6)
X = df[['Average Cost for two','Votes','Price range','Aggregate rating','Longitude','Latitude']]
y = df[['Rating text']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=31)

# Fit the data
knn = knn.fit(X_train, y_train)

# Predict on the data
y_prediction = knn.predict(X_test)

# Classification Report of predictions
print(classification_report(y_test, y_prediction))
print(knn.score(X_test, y_test))


# The accuracy of the above model is 71%. Thus, this data is better suited for KNN classification than linear regression.

# <h2 align ='center' >LOGISTIC REGRESSSION</h2>

# In[13]:

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

dummy8 = pd.get_dummies(df['Rating text'])
df = pd.concat([df,dummy8],axis =1)



# In[15]:


X = df[['Average Cost for two','Votes','Price range','Aggregate rating','Longitude','Latitude']]
y = df[['Excellent']]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)


# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# The accuracy of the above logistic regression model is 97%. Thus, this model is best suited for logistic regression.

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



