
<h1><center>PYTHON ZOMATO PROJECT</h1> 



```python
# Import all the packages and read .csv file from desktop
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/vineet.khattar/Downloads/zomato.csv/zomato.csv',encoding = "ISO-8859-1")
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Restaurant ID</th>
      <th>Restaurant Name</th>
      <th>Country Code</th>
      <th>City</th>
      <th>Address</th>
      <th>Locality</th>
      <th>Locality Verbose</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Cuisines</th>
      <th>...</th>
      <th>Currency</th>
      <th>Has Table booking</th>
      <th>Has Online delivery</th>
      <th>Is delivering now</th>
      <th>Switch to order menu</th>
      <th>Price range</th>
      <th>Aggregate rating</th>
      <th>Rating color</th>
      <th>Rating text</th>
      <th>Votes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6317637</td>
      <td>Le Petit Souffle</td>
      <td>162</td>
      <td>Makati City</td>
      <td>Third Floor, Century City Mall, Kalayaan Avenu...</td>
      <td>Century City Mall, Poblacion, Makati City</td>
      <td>Century City Mall, Poblacion, Makati City, Mak...</td>
      <td>121.027535</td>
      <td>14.565443</td>
      <td>French, Japanese, Desserts</td>
      <td>...</td>
      <td>Botswana Pula(P)</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>3</td>
      <td>4.8</td>
      <td>Dark Green</td>
      <td>Excellent</td>
      <td>314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6304287</td>
      <td>Izakaya Kikufuji</td>
      <td>162</td>
      <td>Makati City</td>
      <td>Little Tokyo, 2277 Chino Roces Avenue, Legaspi...</td>
      <td>Little Tokyo, Legaspi Village, Makati City</td>
      <td>Little Tokyo, Legaspi Village, Makati City, Ma...</td>
      <td>121.014101</td>
      <td>14.553708</td>
      <td>Japanese</td>
      <td>...</td>
      <td>Botswana Pula(P)</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>3</td>
      <td>4.5</td>
      <td>Dark Green</td>
      <td>Excellent</td>
      <td>591</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6300002</td>
      <td>Heat - Edsa Shangri-La</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Edsa Shangri-La, 1 Garden Way, Ortigas, Mandal...</td>
      <td>Edsa Shangri-La, Ortigas, Mandaluyong City</td>
      <td>Edsa Shangri-La, Ortigas, Mandaluyong City, Ma...</td>
      <td>121.056831</td>
      <td>14.581404</td>
      <td>Seafood, Asian, Filipino, Indian</td>
      <td>...</td>
      <td>Botswana Pula(P)</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>4</td>
      <td>4.4</td>
      <td>Green</td>
      <td>Very Good</td>
      <td>270</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6318506</td>
      <td>Ooma</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Third Floor, Mega Fashion Hall, SM Megamall, O...</td>
      <td>SM Megamall, Ortigas, Mandaluyong City</td>
      <td>SM Megamall, Ortigas, Mandaluyong City, Mandal...</td>
      <td>121.056475</td>
      <td>14.585318</td>
      <td>Japanese, Sushi</td>
      <td>...</td>
      <td>Botswana Pula(P)</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>4</td>
      <td>4.9</td>
      <td>Dark Green</td>
      <td>Excellent</td>
      <td>365</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6314302</td>
      <td>Sambo Kojin</td>
      <td>162</td>
      <td>Mandaluyong City</td>
      <td>Third Floor, Mega Atrium, SM Megamall, Ortigas...</td>
      <td>SM Megamall, Ortigas, Mandaluyong City</td>
      <td>SM Megamall, Ortigas, Mandaluyong City, Mandal...</td>
      <td>121.057508</td>
      <td>14.584450</td>
      <td>Japanese, Korean</td>
      <td>...</td>
      <td>Botswana Pula(P)</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>4</td>
      <td>4.8</td>
      <td>Dark Green</td>
      <td>Excellent</td>
      <td>229</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# <font size =4>Create dummy variables from the data</font> 



```python
dummy1 = pd.get_dummies(df['Has Online delivery'])
df = pd.concat([df,dummy1],axis =1)
df.rename(columns={'No':'noonlinedelivery'}, inplace=True)
df.rename(columns={'Yes':'yesonlinedelivery'}, inplace=True)

```


```python
dummy2 = pd.get_dummies(df['Has Table booking'])
df = pd.concat([df,dummy2],axis =1)
df.rename(columns={'No':'Hastable'}, inplace=True)
df.rename(columns={'Yes':'notable'}, inplace=True)

```


```python
dummy3 = pd.get_dummies(df['Is delivering now'])
df = pd.concat([df,dummy3],axis =1)
df.rename(columns={'No':'yesdelivery'}, inplace=True)
df.rename(columns={'Yes':'nodelivery'}, inplace=True)

```


```python
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

```


```python
dummy6 = pd.get_dummies(df['Has Table booking'])
df = pd.concat([df,dummy6],axis =1)
df.rename(columns={'No':'yesorder'}, inplace=True)
df.rename(columns={'Yes':'noorder'}, inplace=True)

```

# <font size =4>Before creating models, I did some EDA on the data<font>


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Restaurant ID</th>
      <th>Country Code</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Average Cost for two</th>
      <th>Price range</th>
      <th>Aggregate rating</th>
      <th>Votes</th>
      <th>noonlinedelivery</th>
      <th>yesonlinedelivery</th>
      <th>...</th>
      <th>Qatar</th>
      <th>Singapore</th>
      <th>South Africa</th>
      <th>Sri Lanka</th>
      <th>Turkey</th>
      <th>UAE</th>
      <th>United Kingdom</th>
      <th>United States</th>
      <th>yesorder</th>
      <th>noorder</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.551000e+03</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>...</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
      <td>9551.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9.051128e+06</td>
      <td>18.365616</td>
      <td>64.126574</td>
      <td>25.854381</td>
      <td>1199.210763</td>
      <td>1.804837</td>
      <td>2.666370</td>
      <td>156.909748</td>
      <td>0.743378</td>
      <td>0.256622</td>
      <td>...</td>
      <td>0.002094</td>
      <td>0.002094</td>
      <td>0.006282</td>
      <td>0.002094</td>
      <td>0.003560</td>
      <td>0.006282</td>
      <td>0.008376</td>
      <td>0.045440</td>
      <td>0.878756</td>
      <td>0.121244</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.791521e+06</td>
      <td>56.750546</td>
      <td>41.467058</td>
      <td>11.007935</td>
      <td>16121.183073</td>
      <td>0.905609</td>
      <td>1.516378</td>
      <td>430.169145</td>
      <td>0.436792</td>
      <td>0.436792</td>
      <td>...</td>
      <td>0.045715</td>
      <td>0.045715</td>
      <td>0.079014</td>
      <td>0.045715</td>
      <td>0.059561</td>
      <td>0.079014</td>
      <td>0.091142</td>
      <td>0.208279</td>
      <td>0.326428</td>
      <td>0.326428</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.300000e+01</td>
      <td>1.000000</td>
      <td>-157.948486</td>
      <td>-41.330428</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.019625e+05</td>
      <td>1.000000</td>
      <td>77.081343</td>
      <td>28.478713</td>
      <td>250.000000</td>
      <td>1.000000</td>
      <td>2.500000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.004089e+06</td>
      <td>1.000000</td>
      <td>77.191964</td>
      <td>28.570469</td>
      <td>400.000000</td>
      <td>2.000000</td>
      <td>3.200000</td>
      <td>31.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.835229e+07</td>
      <td>1.000000</td>
      <td>77.282006</td>
      <td>28.642758</td>
      <td>700.000000</td>
      <td>2.000000</td>
      <td>3.700000</td>
      <td>131.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.850065e+07</td>
      <td>216.000000</td>
      <td>174.832089</td>
      <td>55.976980</td>
      <td>800000.000000</td>
      <td>4.000000</td>
      <td>4.900000</td>
      <td>10934.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>



<h3 align ='center'>Histogram of Aggregated rating </h3>


```python
num_bins = 10
plt.hist(df['Aggregate rating'], num_bins, normed=1, facecolor='blue', alpha=0.5)
plt.xlabel("Rating")
plt.ylabel("Counts")
plt.show()

```


![png](output_11_0.png)


<font>From the histogram we can conclude that majority ofthe rating is spread out between 2 to 5</font>

<h3 align ='center'>Histogram of Number of votes</h3>


```python
num_bins = 10
plt.hist(df['Votes'], num_bins, normed=1, facecolor='blue', alpha=0.5)
plt.xlabel("Votes")
plt.ylabel("Votes")

plt.show()

```


![png](output_14_0.png)


From the above model we can say that majority of votes are betwenn 0-1000 and few between 1000-2000

<h3 align ='center'>SCATTERPLOT</h3>


```python
x = df['Votes']
y =df['Aggregate rating']
colors = (0,0,0)
 
# Plot
plt.scatter(x, y,  c=colors, alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('Votes')
plt.ylabel('Aggregate rating')
plt.show()

```


![png](output_17_0.png)


The above scatter diagram shows the relation between number of votes and aggregating rating. We can see that majority of votes are between 0-2000. Additionally, rating is between 2-5

<h3 align ='center'>LINEAR REGRESSION  </h3>


```python
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


```

    0.688948002449
    0.689618101214
    R^2: 0.32409840220521047
    

<font size =3>We get an accuracy of 32%. Mean square error is 0.68. Error goes down after scaling is done on the data</font>

<h2 align ='center'>KNN CLASSIFICATION</h2>

To get a better accuracy, I perform KNN classification model. It classifies data as per the rating labels (such as Excellent, Average, Good, Very good )


```python
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

```

                 precision    recall  f1-score   support
    
        Average       0.74      0.87      0.80      1112
      Excellent       0.38      0.21      0.27       104
           Good       0.48      0.49      0.49       601
      Not rated       1.00      0.98      0.99       659
           Poor       0.00      0.00      0.00        50
      Very Good       0.46      0.31      0.37       340
    
    avg / total       0.69      0.71      0.69      2866
    
    0.712491277041
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel\__main__.py:11: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    

The accuracy of the above model is 71%. Thus, this data is better suited for KNN classification than linear regression.

<h2 align ='center' >LOGISTIC REGRESSSION</h2>


```python
# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

dummy8 = pd.get_dummies(df['Rating text'])
df = pd.concat([df,dummy8],axis =1)


```


```python

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

```

    [[3699    7]
     [  79   36]]
                 precision    recall  f1-score   support
    
              0       0.98      1.00      0.99      3706
              1       0.84      0.31      0.46       115
    
    avg / total       0.97      0.98      0.97      3821
    
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\utils\validation.py:526: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    

The accuracy of the above logistic regression model is 97%. Thus, this model is best suited for logistic regression.


```python

```


```python

```


```python

```


```python

```
