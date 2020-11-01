import pandas as pd 

data = pd.read_csv('gender-classifier.csv', encoding = 'Latin1')

df = data[['description', 'gender']]

df.isna().sum()
"""
Out[8]: 
description    3744
gender           97
dtype: int64
"""
df.dropna(axis = 0, inplace = True)

df.shape
"""
df.shape
Out[10]: (16224, 2)
"""

# %% Prepare data

df['gender'] = [1 if each == 'male' else 0 for each in df['gender']]
df['gender'].value_counts()
"""
0    10755
1     5469
Name: gender, dtype: int64
"""

import re

example_desc = df.description[13]
"""
Come join the fastest blog network online today http://t.co/S5mFPA1vgK 
and http://t.co/MPUuQtYF1g We cover credit repair, credit cards and bankruptcy.
"""
# There speacial characters that we don't need such as : / 
# We will replace them with spaces.

example_rplc = re.sub('[^a-zA-Z]',' ', example_desc)
"""
Come join the fastest blog network online today http   t co S mFPA vgK and http
   t co MPUuQtYF g We cover credit repair  credit cards and bankruptcy 
"""

# And also Come is different than 'come' etc. 

example_rplc = example_rplc.lower()
"""
come join the fastest blog network online today 
http   t co s mfpa vgk and http   t co mpuuqtyf g we cover credit 
repair  credit cards and bankruptcy
"""

import nltk as nlp
nlp.download('punkt')

example_rplc = nlp.word_tokenize(example_rplc)
# it returns it as a list of strings.
# ex = musn't --> must n't

nlp.download("stopwords")
from nltk.corpus import stopwords

example_rplc = [ word for word in example_rplc if not word in set(stopwords.words("english"))]
# irrelevant words such as 'the' have been removed as they dont have any
# impact for determining.

nlp.download('wordnet')
lemma = nlp.WordNetLemmatizer()
example_rplc = [ lemma.lemmatize(word) for word in example_rplc] 
# meeting => meet

example_rplc = ' '.join(example_rplc)
#make it a sentence again
"""
come join fastest blog network online today http co mfpa vgk http co mpuuqtyf g
cover credit repair credit card bankruptcy
"""
description_cln = []
for description in df.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()   # buyuk harftan kucuk harfe cevirme
    description = nlp.word_tokenize(description)
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_cln.append(description)

# list size = 16624
"""
df.shape
Out[10]: (16224, 2)
"""
# %% Model

from sklearn.feature_extraction.text import CountVectorizer # bag of words 
max_features = 1309

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparse_matrix = count_vectorizer.fit_transform(description_cln).toarray()  # x

print("most used words: {}".format(count_vectorizer.get_feature_names()))

# %%
y = df.iloc[:,1].values   # male or female classes
x = sparse_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.13, random_state = 9)

# %% Modeling

# will be added.




