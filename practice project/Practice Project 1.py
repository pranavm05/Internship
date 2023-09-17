#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier


# In[2]:


rf = RandomForestClassifier()


# In[3]:


df = pd.read_csv("winequality-red.csv")
df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


df.head(45)


# In[9]:


df.tail(45)


# In[10]:


df.shape


# In[11]:


df.columns


# In[12]:


df.columns.tolist()


# In[13]:


df.dtypes


# In[14]:


df.isnull()


# In[15]:


df.isnull().sum()      #this shows that no null values are present in the dataset.


# In[16]:


df.info()


# In[17]:


sns.heatmap(df)


# In[18]:


df['quality'].unique()


# In[19]:


for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# In[20]:


sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x="fixed acidity", bins=30, kde=True)
plt.title("Distribution of Fixed Acidity")
plt.xlabel("Fixed Acidity")
plt.ylabel("Frequency")
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.boxplot(x="quality", y="fixed acidity", data=df)
plt.title("Box Plot of Wine Quality vs. Fixed Acidity")
plt.xlabel("Wine Quality")
plt.ylabel("Fixed Acidity")
plt.show()


# In[22]:


# setting of the cutoff value.

Arb_cutoff = 7


# In[23]:


df['quality_wine']=(df['quality']>=Arb_cutoff).astype(int)


# In[24]:


df


# In[25]:


df.drop('quality',axis=1,inplace=True)


# In[26]:


df


# In[27]:


# As our target variable 'y' is having 2 variables that is 0 to 1 decision tree can be a fit model.


# In[28]:


x=df.drop('quality_wine', axis=1)
y=df['quality_wine']


# In[29]:


x


# In[30]:


from sklearn.ensemble import RandomForestClassifier


# In[31]:


rf = RandomForestClassifier()


# In[32]:


rf.fit(x,y)


# In[33]:


rf


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# In[42]:


print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[43]:


rf = RandomForestClassifier(random_state=42)


# In[46]:


rf.fit(x_train,y_train)


# In[47]:


y_pred = rf.predict(x_test)


# In[48]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[50]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[51]:


report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[52]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


# In[55]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# In[57]:


y_pred = rf.predict(x_test)


# In[58]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[59]:


precision = precision_score(y_test, y_pred)
print("Precision:", precision)


# In[60]:


recall = recall_score(y_test, y_pred)
print("Recall:", recall)


# In[61]:


f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


# In[64]:


y_pred_proba = rf.predict_proba(x_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




