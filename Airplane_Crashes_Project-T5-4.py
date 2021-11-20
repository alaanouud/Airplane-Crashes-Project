#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

from collections import defaultdict
from collections import Counter

import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import (LSTM, 
                          Embedding, 
                          BatchNormalization,
                          Dense, 
                          TimeDistributed, 
                          Dropout, 
                          Bidirectional,
                          Flatten, 
                          GlobalMaxPool1D)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# In[18]:


df=pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')
df.head(5)


# In[19]:


df['Year'] = pd.DatetimeIndex(df['Date']).year
df['Year'].head(5)


# # Data Cleaning

# In[20]:


df.info()


# In[21]:


df.columns


# In[22]:


df.isna().sum()


# In[23]:


data = df.dropna()
data.head(3)


# In[24]:


print('number of rows  before discarding duplicates = %d' % (data.shape[0]))
airplane= data.drop_duplicates()
print("number of rows after discarding duplicates = %d" % (data.shape[0]))


# In[25]:


airplane.isnull().values.any()


# # EDA

# In[26]:


Crshs=pd.DataFrame(airplane.Year.value_counts())

plt.figure(figsize=(15, 5))
plt.bar(x=Crshs.index, height=Crshs["Year"])
plt.title("Number of Crashes each Year")
plt.show()


# In[27]:


airplane['Location'].value_counts()[:10].plot(kind='barh'
                                        ,  figsize=(20,10), rot=20, fontsize=22, title="Top 15 countries by crash locations")


# In[28]:


fatals=pd.DataFrame(airplane.Fatalities.groupby(airplane.Year).sum())

plt.figure(figsize=(10, 10))
plt.bar(x=fatals.index, height=fatals["Fatalities"])
plt.title("Number of People died in Plane each Year")
plt.show()


# In[29]:


airplane['Survivor']=airplane['Aboard']-airplane['Fatalities']
airplane.head(2)


# In[30]:


FSG_per_year = airplane[['Year', 'Fatalities', 'Survivor', 'Ground']].groupby('Year').sum()
FSG_per_year = FSG_per_year.reset_index()


# In[31]:


sns.lineplot(x = 'Year', y = 'Fatalities', data = FSG_per_year, color = 'green')
sns.lineplot(x = 'Year', y = 'Survivor', data = FSG_per_year, color = 'blue')
sns.lineplot(x = 'Year', y = 'Ground', data = FSG_per_year, color = 'red')
plt.legend(['Fatalities', 'Survival', 'Ground'])
plt.xlabel('Years')
plt.ylabel('Count')


# In[32]:


Locations = airplane.groupby('Location', as_index=False).agg({'Year':'mean'}).sort_values('Year', ascending=False)


# In[33]:


fig = px.choropleth(Locations, 
                    locations = 'Location', 
                    locationmode = 'country names', 
                    color = 'Year',
                    hover_data = ['Year'], 
                    title = 'Most Dangerous Locations per years')
fig.show()


# In[34]:


SurvivorByPlaneType=pd.DataFrame(airplane.Survivor.groupby(airplane.Type).sum())
SurvivorByPlaneType = SurvivorByPlaneType.sort_values(by='Survivor', ascending=False)
SurvivorByPlaneType.head(5)


# In[35]:


SurvivorByPlaneType[:5].plot(kind='barh'
                                        ,  figsize=(20,10), rot=20, fontsize=22, title="Top 5 Airplane Type has number of Survivor")


# In[36]:


CrashesByPlaneType=pd.DataFrame(airplane.Fatalities.groupby(airplane.Type).sum())
CrashesByPlaneType = CrashesByPlaneType.sort_values(by='Fatalities', ascending=False)
CrashesByPlaneType.head(5)


# In[37]:


CrashesByPlaneType[:5].plot(kind='barh'
                                        ,  figsize=(20,10), rot=20, fontsize=22, title="Top 5 Airplane Type has number of Fatalities")


# In[38]:


Opreator_total = airplane.groupby('Operator')[['Operator']].count()
Opreator_total = Opreator_total.rename(columns={"Operator": "Count"}) 
Opreator_total = Opreator_total.sort_values(by='Count', ascending=False).head(10)

plt.figure(figsize=(12,6))
sns.barplot(y=Opreator_total.index, x="Count", data=Opreator_total,palette="Blues_d",orient='h')
plt.xlabel('Count', fontsize=10)
plt.ylabel('Operator', fontsize=10)
plt.title('Total Count by Opeartor', loc='Center', fontsize=14)
plt.show()


# # Data Pre-processing

# In[ ]:


import re
import string

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text
airplane['Summary_clean'] = airplane['Summary'].apply(clean_text)
airplane.head()

# StopWords

from nltk.corpus import stopwords
data_text = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
data_text = data_text + more_stopwords

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in data_text)
    return text
    
airplane['Summary_clean'] = airplane['Summary_clean'].apply(remove_stopwords)
airplane.head(2)


# In[41]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wc = WordCloud(
    background_color='white', 
    max_words=200, 
)
wc.generate(' '.join(text for text in airplane['Summary']))
plt.figure(figsize=(15,10))
plt.title('Top words for Crashed Airplanes Summary', 
          fontdict={'size': 20,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[42]:


airplane['Summary_clean'].nunique() 


# # Kmeans Cluster

# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[44]:


documents = list(airplane['Summary_clean'])
vectorizer = TfidfVectorizer(stop_words='english') # Stop words are like "a", "the", or "in" which don't have significant meaning
X = vectorizer.fit_transform(documents)


# In[45]:


model = MiniBatchKMeans(n_clusters=5, random_state=100)
model.fit(X)


# In[46]:


model.cluster_centers_


# In[47]:


model.predict(X)
model.labels_


# In[48]:


print ('Most Common Terms per Cluster:')

order_centroids = model.cluster_centers_.argsort()[:,::-1] #sort cluster centers by proximity to centroid
terms = vectorizer.get_feature_names()

for i in range(5):
    print("\n")
    print('Cluster %d:' % i)
    for j in order_centroids[i, :10]: #replace 10 with n words per cluster
        print ('%s' % terms[j]),
    print


# In[49]:


pca = PCA(n_components=13, random_state=100)
reduced_features = pca.fit_transform(X.toarray())

# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(model.cluster_centers_)


# In[50]:


plt.scatter(reduced_features[:,0], reduced_features[:,1], c=model.predict(X))
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')


# # Modeling

# In[51]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split


# # NMF

# In[52]:


nltk.download('punkt')


# In[53]:


df=airplane[['Summary_clean','Operator','Location']]


# In[54]:


pd.set_option('display.max_colwidth',-1)
df.head(2)


# In[55]:


X_train, X_test = train_test_split(df, test_size=0.3, random_state=100)


# In[56]:


X_train['Operator'].value_counts()


# In[57]:


stemmer = nltk.stem.SnowballStemmer('english')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))


# In[58]:


def tokenize(text):
    tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('Xx/')) > 2 and len(re.sub('\d+', '', word.strip('Xx/'))) > 3) ] 
    tokens = map(str.lower, tokens)
    stems = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    return stems


# In[59]:


# Instatiate
vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=None, max_df=0.75, max_features=1000, lowercase=False, ngram_range=(1,2))
# Fit & Transform
tfidf_vectors = vectorizer.fit_transform(X_train['Summary_clean'].values.astype('U')) ## Even astype(str) would work


# In[60]:


tfidf_vectors.shape 


# In[61]:


tfidf_vectors.A


# In[62]:


len(vectorizer.get_feature_names())


# In[63]:


clf = decomposition.NMF(n_components=6, random_state=111) # components is the number of topics

W1 = clf.fit_transform(tfidf_vectors)
H1 = clf.components_


# In[64]:


# NMF Decomposition
tfidf_vectors.shape


# In[65]:


W1.shape


# In[66]:


W1


# In[67]:


H1.shape


# In[68]:


H1


# In[69]:


H1[:, 0:1]


# In[70]:


num_words=10 # TOPIC IS DEFINED AS A COLLECTION OF 15 WORDS

vocab = np.array(vectorizer.get_feature_names())

top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_words-1:-1]]
topic_words = ([top_words(t) for t in H1])
topics = [' '.join(t) for t in topic_words]


# In[71]:


topics


# In[72]:


colnames =["Topic" + str(i) for i in range(clf.n_components)]
docnames = ["Doc" + str(i) for i in range(len(X_train.Summary_clean))]
df_doc_topic = pd.DataFrame(np.round(W1, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic
df_doc_topic.rename(columns = {"Topic0":"Weather","Topic1":"Fire","Topic2":"Bomb","Topic3":"Failur"
                               ,"Topic4":"Mountain","Topic5":"Crash"} ,inplace = False)


# In[73]:


X_train.head()


# In[74]:


Wtest = clf.transform(vectorizer.transform(X_test.Summary_clean[:10]))


# In[75]:


colnames = ["Topic" + str(i) for i in range(clf.n_components)]
docnames = ["Doc" + str(i) for i in range(len(X_test[:10].Summary_clean))]
df_doc_topic = pd.DataFrame(np.round(Wtest, 3), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic
df_doc_topic.rename(columns = {"Topic0":"Weather","Topic1":"Fire","Topic2":"Bomb","Topic3":"Failur"
                               ,"Topic4":"Mountain","Topic5":"Crash"} ,inplace = False)


# # LDA

# In[76]:


vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=None, max_df=0.75, max_features=1000, lowercase=False, ngram_range=(1,2))

tf_vectors = vectorizer.fit_transform(X_train['Summary_clean'].values.astype('U'))


# In[77]:


lda = decomposition.LatentDirichletAllocation(n_components=3, max_iter=5, learning_method='online', learning_offset=50, n_jobs=-1, random_state=100)

W1 = lda.fit_transform(tf_vectors)
H1 = lda.components_


# In[78]:


W1


# In[79]:


W1.shape


# In[80]:


num_words=10

vocab = np.array(vectorizer.get_feature_names())

top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_words-1:-1]]
topic_words = ([top_words(t) for t in H1])
topics = [' '.join(t) for t in topic_words]


# In[81]:


len(topics)


# In[82]:


topics


# In[83]:


colnames = ["Topic" + str(i) for i in range(lda.n_components)]
docnames = ["Doc" + str(i) for i in range(len(X_train.Summary_clean))]
df_doc_topic = pd.DataFrame(np.round(W1, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic
df_doc_topic.rename(columns = {"Topic0":"Weather","Topic1":"Fire","Topic2":"Crash","Topic3":"Failur"} ,inplace = False)


# In[84]:


X_train.head()


# In[85]:


Wtest = lda.transform(vectorizer.transform(X_test.Summary_clean[:5]))


# In[86]:


Wtest.shape


# In[87]:


colnames = ["Topic" + str(i) for i in range(lda.n_components)]
docnames = ["Doc" + str(i) for i in range(len(X_test.Summary_clean[:5]))]
df_doc_topic = pd.DataFrame(np.round(Wtest, 2), columns=colnames, index=docnames)
significant_topic = np.argmax(df_doc_topic.values, axis=1)
df_doc_topic['dominant_topic'] = significant_topic
df_doc_topic.rename(columns = {"Topic0":"Weather","Topic1":"Fire","Topic2":"Crash","Topic3":"Failur"} ,inplace = False)


# In[88]:


df_doc_topic.shape


# In[89]:


X_test.shape


# In[90]:


X_test.head()


# # SVD

# In[91]:


example=airplane["Summary_clean"]


# In[92]:


vectorizer = CountVectorizer(stop_words = 'english')
doc_word = vectorizer.fit_transform(example)
doc_word.shape


# In[93]:


pd.DataFrame(doc_word.toarray(), index=example, columns=vectorizer.get_feature_names()).head(2)


# In[94]:


from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD

lsa = TruncatedSVD(2)
# Fit & Transform
doc_topic = lsa.fit_transform(doc_word)


# In[95]:


print (doc_topic)


# In[96]:


import numpy as np
pd.DataFrame(np.round((doc_topic),4))


# In[97]:


print (lsa.explained_variance_ratio_)


# In[98]:


import numpy as np
np.diag(lsa.explained_variance_ratio_)
pd.DataFrame(np.diag(lsa.explained_variance_ratio_))


# In[99]:


topic_word = pd.DataFrame(lsa.components_.round(3),
             index = ["component_1","component_2"], 
             columns = vectorizer.get_feature_names())
topic_word


# In[100]:


def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


# In[101]:


display_topics(lsa, vectorizer.get_feature_names(), 7)


# In[102]:


Vt = pd.DataFrame(doc_topic.round(10),
             index = example,
             columns = ["component_1","component_2" ])
Vt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




