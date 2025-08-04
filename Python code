```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

sns.set(style='whitegrid')

```


```python
# Αν το αρχείο βρίσκεται στον ίδιο φάκελο
df = pd.read_csv("heart.csv")
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPainType</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>RestingECG</th>
      <th>MaxHR</th>
      <th>ExerciseAngina</th>
      <th>Oldpeak</th>
      <th>ST_Slope</th>
      <th>HeartDisease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>289</td>
      <td>0</td>
      <td>Normal</td>
      <td>172</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>F</td>
      <td>NAP</td>
      <td>160</td>
      <td>180</td>
      <td>0</td>
      <td>Normal</td>
      <td>156</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>283</td>
      <td>0</td>
      <td>ST</td>
      <td>98</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>F</td>
      <td>ASY</td>
      <td>138</td>
      <td>214</td>
      <td>0</td>
      <td>Normal</td>
      <td>108</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>195</td>
      <td>0</td>
      <td>Normal</td>
      <td>122</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

```

    (918, 12)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 918 entries, 0 to 917
    Data columns (total 12 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Age             918 non-null    int64  
     1   Sex             918 non-null    object 
     2   ChestPainType   918 non-null    object 
     3   RestingBP       918 non-null    int64  
     4   Cholesterol     918 non-null    int64  
     5   FastingBS       918 non-null    int64  
     6   RestingECG      918 non-null    object 
     7   MaxHR           918 non-null    int64  
     8   ExerciseAngina  918 non-null    object 
     9   Oldpeak         918 non-null    float64
     10  ST_Slope        918 non-null    object 
     11  HeartDisease    918 non-null    int64  
    dtypes: float64(1), int64(6), object(5)
    memory usage: 86.2+ KB
    None
                  Age   RestingBP  Cholesterol   FastingBS       MaxHR  \
    count  918.000000  918.000000   918.000000  918.000000  918.000000   
    mean    53.510893  132.396514   198.799564    0.233115  136.809368   
    std      9.432617   18.514154   109.384145    0.423046   25.460334   
    min     28.000000    0.000000     0.000000    0.000000   60.000000   
    25%     47.000000  120.000000   173.250000    0.000000  120.000000   
    50%     54.000000  130.000000   223.000000    0.000000  138.000000   
    75%     60.000000  140.000000   267.000000    0.000000  156.000000   
    max     77.000000  200.000000   603.000000    1.000000  202.000000   
    
              Oldpeak  HeartDisease  
    count  918.000000    918.000000  
    mean     0.887364      0.553377  
    std      1.066570      0.497414  
    min     -2.600000      0.000000  
    25%      0.000000      0.000000  
    50%      0.600000      1.000000  
    75%      1.500000      1.000000  
    max      6.200000      1.000000  
    Age               0
    Sex               0
    ChestPainType     0
    RestingBP         0
    Cholesterol       0
    FastingBS         0
    RestingECG        0
    MaxHR             0
    ExerciseAngina    0
    Oldpeak           0
    ST_Slope          0
    HeartDisease      0
    dtype: int64
    


```python
# Κατανομή στόχου
sns.countplot(x='HeartDisease', data=df)
plt.title('Κατανομή Καρδιακής Νόσου')
plt.show()

# Θερμικός χάρτης συσχέτισης
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Συσχέτιση Μεταβλητών')
plt.show()


```


    
![png](output_3_0.png)
    



    
![png](output_3_1.png)
    



```python
# Χωρισμός χαρακτηριστικών και στόχου
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# One-hot encoding (αν υπάρχουν κατηγορικές)
X = pd.get_dummies(X, drop_first=True)

# Κανονικοποίηση αριθμητικών μεταβλητών
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

```


```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

```


```python
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Logistic Regression:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

```

    Logistic Regression:
    [[67 10]
     [17 90]]
                  precision    recall  f1-score   support
    
               0       0.80      0.87      0.83        77
               1       0.90      0.84      0.87       107
    
        accuracy                           0.85       184
       macro avg       0.85      0.86      0.85       184
    weighted avg       0.86      0.85      0.85       184
    
    


```python
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest:")
print(classification_report(y_test, rf_pred))

```

    Random Forest:
                  precision    recall  f1-score   support
    
               0       0.83      0.82      0.82        77
               1       0.87      0.88      0.87       107
    
        accuracy                           0.85       184
       macro avg       0.85      0.85      0.85       184
    weighted avg       0.85      0.85      0.85       184
    
    


```python
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

```


    
![png](output_8_0.png)
    


    ROC AUC Score: 0.9268115062507586
    

## Συμπεράσματα:

- Η Logistic Regression δείχνει 0.85 % ακρίβεια, ενώ η Random Forest έδωσε παρόμοια απόδοση.
- Η ηλικία, το επίπεδο χοληστερόλης και η αρτηριακή πίεση φαίνεται να έχουν ισχυρή συσχέτιση με την καρδιακή νόσο.
- Θα μπορούσε να βελτιωθεί η απόδοση με hyperparameter tuning ή επιπλέον χαρακτηριστικά.



```python

```
