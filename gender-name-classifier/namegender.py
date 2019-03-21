
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import requests
from bs4 import BeautifulSoup
import matplotlib as mlp
mlp.use("TKAgg")
'exec(%matplotlib inline)'
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
mlp.rcParams.update({'font.family': "Open Sans", 'font.size' : 16})

#import social security names database from Kaggle in int form
names = pd.read_csv("input/NationalNames.csv", sep=',', dtype = {'Count': np.int32})
names = names.fillna(0)

# list the first few lines of csv file
#print(names.head())

# Groupby
namechart = names.groupby(['Name', 'Gender'], as_index=False)['Count'].sum()
#print(namechart.head(5))


# Add a column to categorize different names into a male 
# or female bucket based on whether or not the frequency
# of males for a name outnambers the frequency o females
namechartdiff = namechart.reset_index().pivot('Name', 'Gender', 'Count')
namechartdiff = namechartdiff.fillna(0)
namechartdiff["Mpercent"] = ((namechartdiff["M"] - namechartdiff["F"])/(namechartdiff["M"] + namechartdiff["F"]))
namechartdiff['gender'] = np.where(namechartdiff['Mpercent'] > 0.001, 'male', 'female')

#print(namechartdiff.head())


# Break down the string of names into bigram blocks of
# characters with CountVectorizer        [no idea why]
char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
X = char_vectorizer.fit_transform(namechartdiff.index)
X = X.tocsc()
y = (namechartdiff.gender == 'male').values.astype(np.int)
#print(X)

# Split our training and test data now
itrain, itest = train_test_split(range(namechartdiff.shape[0]), train_size=0.7)
mask=np.ones(namechartdiff.shape[0], dtype='int')
mask[itrain]=1
mask[itest]=0
mask = (mask==1)

# Train the model
Xtrainthis=X[mask]
Ytrainthis=y[mask]
Xtestthis=X[~mask]
Ytestthis=y[~mask]
clf = MultinomialNB(alpha = 1)
clf.fit(Xtrainthis, Ytrainthis)
training_accuracy = clf.score(Xtrainthis,Ytrainthis)
test_accuracy = clf.score(Xtestthis,Ytestthis)

print('Training accuracy:')
print(training_accuracy)
print('Test     accuracy:')
print(test_accuracy)

# Function
def lookup(x):
    str(x)
    new = char_vectorizer.transform([x])
    y_pred = clf.predict(new)
    if (y_pred == 1):
        print("This is most likely a male name!")
    else:
        print("This is most likely a female name!")


nametocheck = 'Roger'
lookup(nametocheck)
