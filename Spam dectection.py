
# coding: utf-8

# ### Importing the required modules/packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


# ### Loading file and looking into the dimensions of data

# In[2]:


raw_data = pd.read_csv("SMSSpamCollection.tsv",sep='\t',names=['label','text'])
pd.set_option('display.max_colwidth',100)
raw_data.head()


# In[18]:


print(raw_data.shape)
pd.crosstab(raw_data['label'],columns = 'label',normalize=True)

