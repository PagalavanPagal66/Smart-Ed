#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[2]:


df = pd.read_excel(r"C:\Users\pagal\OneDrive\Desktop\extra\Dataset-iTech - backup.xlsx")


# In[3]:


df


# In[4]:


df.dropna(inplace = True)


# In[5]:


df


# In[6]:


pd.DataFrame(df.Cluster.unique())


# In[7]:


df['category_id'] = df['Cluster'].factorize()[0]
category_id_df = df[['Cluster', 'category_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Cluster']].values)

# New dataframe
df


# In[8]:


df['Details'][0]


# In[9]:


fig = plt.figure(figsize=(8,6))
colors = ['grey','grey','grey','grey','grey','grey','grey','grey',
    'grey','darkblue','darkblue','darkblue','red','red','red','green','green','green','green','yellow','yellow','yellow','yellow']
df.groupby('Cluster').Details.count().sort_values().plot.barh(
    ylim=0, color=colors, title= 'NUMBER OF COMPLAINTS IN EACH PRODUCT CATEGORY\n')
plt.xlabel('Number of ocurrences', fontsize = 10);


# In[10]:


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

# We transform each complaint into a vector
features = tfidf.fit_transform(df.Details).toarray()

labels = df.category_id

print("Each of the %d video details is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))

features


# In[11]:


X = df['Details']
y = df['Cluster'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)


# In[12]:


models = [
    RandomForestClassifier(),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=63),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[13]:


mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc


# In[14]:


plt.figure(figsize=(8,5))
sns.boxplot(x='model_name', y='accuracy', 
            data=cv_df, 
            color='lightblue', 
            showmeans=True)
plt.title("MEAN ACCURACY (cv = 5)\n", size=14);


# In[15]:


X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df.index, test_size=0.2, 
                                                               random_state=36)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(features)
y_test.unique()
y_train.unique()
model.score(X_train,y_train)
model.score(X_test,y_test)


# In[16]:


print('\t\t\t\tCLASSIFICATIION METRICS\n')
print(metrics.classification_report(labels, y_pred, target_names= df['Cluster'].unique()))


# In[17]:


conf_mat = confusion_matrix(labels, y_pred)
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(conf_mat, annot=True, cmap="Greens", fmt='d',
            xticklabels=category_id_df.Cluster.values, 
            yticklabels=category_id_df.Cluster.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=35);


# In[ ]:




