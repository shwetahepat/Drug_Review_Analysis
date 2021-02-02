# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:13:26 2021

@author: Ankit
"""

#DRUG Reviwe Analysis


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from mlxtend.plotting import plot_decision_regions
from wordcloud import WordCloud
from wordcloud import STOPWORDS

df_train = pd.read_csv("C://Users//Ankit//Documents//1st CDAC Shweta//Machine Learning//Programs//drugsComTrain_raw.tsv", sep='\t')
df_test = pd.read_csv("C://Users//Ankit//Documents//1st CDAC Shweta//Machine Learning//Programs//drugsComTest_raw.tsv", sep='\t')

print(df_train.head(n=10))
print("Shape of train :", df_train.shape) #Shape of train : (161297, 7)
print("Shape of test :", df_test.shape) #Shape of test : (53766, 7)

#Data cleaning and EDA
#***************************

del df_train['uniqueid']
del df_train['date']


#Data cleaning and EDA
#printing unique no of drugs and unique no of condition
print("number of drugs:", len(df_train['drugName'].unique()))#3436
print("number of conditions:", len(df_train['condition'].unique()))#885


# Calculating how many drugs are there for per/each condition
drug_per_condition = df_train.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
print(drug_per_condition)
print(drug_per_condition[:30]) #displaying upto 30 conditions.
drug_per_condition.shape

#as there is wrong name of condition like  <span> so we have to replace it by NAN
#replace wrong name in condition column with NaN

#Replacing with NAN  -- 3</span>
df_train.loc[df_train['condition'].str.contains('</span>',case=False, na=False), 'condition'] = 'NAN'
print(df_train[:30])
df_train['condition'].replace('NAN', np.NaN, inplace=True)
df_train['condition'].replace('Not Listed / Othe', np.NaN, inplace=True)


#create a dictionary with drugname:condition to fill NaN
dictionary=df_train.set_index('drugName')['condition'].to_dict()
len(dictionary)


#fill NaN value with correct condition names using created dictionaryC://Users//Ankit//Documents//1st CDAC Shweta//Machine Learning//Programs//C://Users//Ankit//Documents//1st CDAC Shweta//Machine Learning//Programs//
df_train.condition.fillna(df_train.drugName.map(dictionary), inplace=True)
df_train.info()

#drop rows with still missing values in condition (100 rows = 0.0006% of total data)
df_train.dropna(inplace=True)

#after cleaning
drug_per_condition = df_train.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
print(drug_per_condition)
print(drug_per_condition[:30])

#PLOTS :

#creating plot to check  Top 10 drug based on condition
drug_per_condition[:10].plot(kind="bar", figsize = (14,6), fontsize = 10, color="#B2B2D8")
plt.xlabel("", fontsize = 20)
plt.ylabel("", fontsize = 20)
plt.title("Top 10 Number of Drugs / Condition", fontsize = 20)
#plt.savefig('C:\Users\kunal')

###########################

#Top20 :  number of drugs per condition.
condition_dn = df_train.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
condition_dn[0:20].plot(kind="bar", figsize = (14,6), fontsize = 10,color="green")
plt.xlabel("", fontsize = 20)
plt.ylabel("", fontsize = 20)
plt.title("Top20 :  number of drugs per condition.", fontsize = 20)

from collections import defaultdict
df_all_6_10 = df_train[df_train["rating"]>5]
df_all_1_5 = df_train[df_train["rating"]<6]

#plotting rating count values :
rating = df_train['rating'].value_counts().sort_values(ascending=False)
rating.plot(kind="bar", figsize = (14,6), fontsize = 10,color="green")
plt.xlabel("", fontsize = 20)
plt.ylabel("", fontsize = 20)
plt.title("Count of rating values", fontsize = 20)





#select conditions with less than 11 drugs
condition_1=drug_per_condition[drug_per_condition<=10].keys()
print(condition_1)
condition_1.shape

#selecting all the drugs where condition with less than 11 drugs is not there 
df_train1=df_train[~df_train['condition'].isin(condition_1)]
df_train1.info()


condition_list=df_train1['condition'].tolist()
corpus_train=df_train1.review

#NLP :
    
import re # Regular expression library
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

#we perform stemming and lemmatizing to get meaningfull wordson the reviwes
stop = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
import spacy
##nlp = spacy.load("en_core_web_sm")

#remove words needs for sentiment analysis from stopwords
n = ["aren't","couldn't","didn't","doesn't","don't","hadn't",
     "hasn't","haven't","isn't","mightn't","mustn't","needn't"
     ,"no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
for i in n:
    stop.remove(i)

#add more words to stopwords
a = ['mg', 'week', 'month', 'day', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 
     'august', 'september','october','november','december', 'iv','oral','pound', 'lb', 'month', 'day','night']
for j in a:
    stop.add(j)



# Text preprocessing steps - remove numbers, captial letters and punctuation -- creating lambda function
alphanumeric=lambda x: re.sub('[^a-zA-Z]', ' ', str(x) )#selecting only a-zA-Z 

punc_lower=lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower()) # selecting puctuation marks and lower case
split=lambda x: x.split()

df_train1['review'] = df_train1.review.map(alphanumeric).map(punc_lower).map(split)
print(df_train1)

#remove stopwords

df_train1['review_clean']=df_train1['review'].apply(lambda x: [item for item in x if item not in stop])

#lemmatizing
#converting multiple similar meaning words into one single root word.
df_train1['review_lemm']=df_train1['review_clean'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

#remove repetative review column
del df_train1['review']
del df_train1['review_clean']


df_train1['review']=df_train1['review_lemm'].apply(lambda x:' '.join(x))

del df_train1['review_lemm']
print(df_train1)


#now same data leaning we have to do with test dataset
df_test.info()

print("number of drugs:", len(df_test['drugName'].unique()))
print("number of conditions:", len(df_test['condition'].unique()))

#delete condition with less than 11 drugs
df_test1=df_test[~df_test['condition'].isin(condition_1)]
df_test1.info()

#delete condition with less than 11 drugs
df_test1=df_test[~df_test['condition'].isin(condition_1)]
df_test1.info()

df_test1.dropna(inplace=True)

# Text preprocessing steps - remove numbers, captial letters and punctuation
alphanumeric=lambda x: re.sub('[^a-zA-Z]', ' ', x)
punc_lower=lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
split=lambda x: x.split()

df_test1['review'] = df_test1.review.map(alphanumeric).map(punc_lower).map(split)

#remove stopwords
df_test1['review_clean']=df_test1['review'].apply(lambda x: [item for item in x if item not in stop])
#lemmatizing
df_test1['review_lemm']=df_test1['review_clean'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
del df_test1['review']
del df_test1['review_clean']

df_test1['review']=df_test1['review_lemm'].apply(lambda x:' '.join(x))
  
del df_test1['review_lemm']
print(df_test1)  

#SENTIMENT MODELING :
    
#drug_recommendation-topic_modeling--Sentiment 
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities, matutils
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity  

df_train1.info()

#drop Nan value
df_train1.dropna(inplace=True)

#drop Nan value
df_train1.dropna(inplace=True)
df_train1['condition']

condition_list=df_train1['condition'].tolist()
corpus_train=df_train1.review
# corpus_test=df_test_s.review
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
n = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't",
     "mightn't","mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
for i in n:
    stop.remove(i)

a = ['mg', 'week', 'month', 'day', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 
     'august', 'september','october','november','december', 'iv','oral','pound',]
for j in a:
    stop.add(j)

#CountVectorizer
# Create a CountVectorizer for parsing/counting words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(2, 2), min_df=10, max_df=0.8)
cv.fit(corpus_train)
doc_word = cv.transform(corpus_train).transpose()
#pd.DataFrame(doc_word.toarray(), cv.get_feature_names()).head()

corpus = matutils.Sparse2Corpus(doc_word)
id2word = dict((v, k) for k, v in cv.vocabulary_.items())

len(id2word)

#Latent Dirichlet Allocation (LDA)

lda = models.LdaModel(corpus=corpus, num_topics=2, id2word=id2word, passes=10)
lda.print_topics()
lda_corpus = lda[corpus]
lda_corpus
lda_docs = [doc for doc in lda_corpus]
lda_docs[0:5]
len(lda_docs)

from wordcloud import WordCloud
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda.show_topics(formatted=False)

fig, axes = plt.subplots(1, 2, figsize=(10,20), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/wc_bigram_lda-2.svg')

lda1 = models.LdaModel(corpus=corpus, num_topics=4, id2word=id2word, passes=10)
lda1.print_topics()


all_topics = lda1.get_document_topics(corpus)
all_topics


num_docs = len(all_topics)
num_docs
num_topics=4

lda_scores = np.empty([num_docs, num_topics])
print(lda_scores.shape)

for i in range(0, num_docs):
    lda_scores[i] = np.array(all_topics[i]).transpose()[1]

lda_corpus1 = lda1[corpus]
lda_corpus1

lda_docs1 = [doc for doc in lda_corpus1]

lda_docs1[0:5]

len(lda_docs1)

def dominant_topic(ldamodel, corpus, texts):
     #Function to find the dominant topic in each review
     sent_topics_df = pd.DataFrame() 
     # Get main topic in each review
     for i, row in enumerate(ldamodel[corpus]):
         row = sorted(row, key=lambda x: (x[1]), reverse=True)
         # Get the Dominant topic, Perc Contribution and Keywords for each review
         for j, (topic_num, prop_topic) in enumerate(row):
             if j == 0:  # => dominant topic
                 wp = ldamodel.show_topic(topic_num,topn=4)
                 topic_keywords = ", ".join([word for word, prop in wp])
                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
             else:
                 break
     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
     contents = pd.Series(texts)
     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
     return(sent_topics_df)

df_dominant_topic = dominant_topic(ldamodel=lda1, corpus=corpus, texts=df_train1['review']) 
df_dominant_topic.head()

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics1 = lda1.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words1 = dict(topics1[i][1])
    cloud.generate_from_frequencies(topic_words1, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/wc_bigram_lda-4.svg')

lda2 = models.LdaModel(corpus=corpus, num_topics=6, id2word=id2word, passes=10)
lda2.print_topics()
lda_corpus2 = lda2[corpus]
lda_corpus2
lda_docs2 = [doc for doc in lda_corpus2]
lda_docs2[0:5]

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics2 = lda2.show_topics(formatted=False)

fig, axes = plt.subplots(2, 3, figsize=(12,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words2 = dict(topics2[i][1])
    cloud.generate_from_frequencies(topic_words2, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/wc_bigram_lda-6.svg')
#drug_recommendation-top_10-topics


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

plt.style.use('ggplot')
#%config InlineBackend.figure_format = 'svg'
#%matplotlib inline
np.set_printoptions(suppress=True) # Suppress scientific notation where possible



from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, make_scorer
from sklearn.datasets import fetch_20newsgroups
import gensim




df_train1.info()
df_train1.dropna()


df_dominant_topic.info()
df_dominant_topic.shape


df = pd.concat([df_train1, df_dominant_topic], axis=1, join='inner')
del df['review']

df


drug_per_condition = df.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
drug_per_condition

drug_per_condition[:10]

condition_1=drug_per_condition[:10].keys()
condition_1

#selecting only top 10 conditions
df_top_10=df[df['condition'].isin(condition_1)]
df_top_10.head()


top_10=df_top_10.groupby(['condition']).Dominant_Topic.value_counts(normalize=True)
top_10


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

plt.style.use('ggplot')
#%config InlineBackend.figure_format = 'svg'
#%matplotlib inline
np.set_printoptions(suppress=True) # Suppress scientific notation where possible





#supervised Modeling:-----
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, make_scorer
from sklearn.datasets import fetch_20newsgroups
import gensim              

                                   

                        
from wordcloud import WordCloud

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(10,8), 
                   title = None, title_size=40, image_color=False):

    wordcloud = WordCloud(background_color='white',
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
        
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()

top_words=df_train1.review.value_counts(normalize=True)[:40].keys()                               



plot_wordcloud(top_words)
#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/wordcloud.svg')

df_train1.rating.value_counts(normalize=True)

# Remove 4-7 star reviews
df_train2 = df_train1.drop(df_train1[(df_train1['rating'] > 4.0) & (df_train1['rating'] < 6.0)].index)

# Set 8-10 star reviews to positive(1), the rest to negative(0)
df_train2['sentiment'] = np.where(df_train2['rating'] >= 7, '1', '0')

df_train2


# Remove 4-7 star reviews
df_test2 = df_test1.drop(df_test1[(df_test1['rating'] > 4.0) & (df_test1['rating'] < 6.0)].index)

# Set 8-10 star reviews to positive(1), the rest to negative(0)
df_test2['sentiment'] = np.where(df_test2['rating'] >= 7, '1', '0')



# Note that the dataset has mostly positive reviews
df_train2.sentiment.value_counts(normalize=True)

sentiment1 = df_train2['sentiment'].value_counts().sort_values(ascending=False)
sentiment1.plot(kind="bar", figsize = (12,6), fontsize = 10,color="green")
plt.xlabel("", fontsize = 20)
plt.ylabel("", fontsize = 20)



X_train=df_train2.review
y_train=df_train2.sentiment
X_test=df_test2.review
y_test=df_test2.sentiment

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
n = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't",
     "mightn't","mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
for i in n:
    stop.remove(i)

a = ['mg', 'week', 'month', 'day', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 
     'august', 'september','october','november','december', 'iv','oral','pound',]
for j in a:
    stop.add(j)



from sklearn.feature_extraction.text import CountVectorizer
#*unigram
cv1 = CountVectorizer(stop_words=stop, ngram_range=(1, 1), min_df=10, max_df=0.7)

X_train_cv1 = cv1.fit_transform(X_train)
X_test_cv1  = cv1.transform(X_test)

# The second document-term matrix has both unigrams and bigrams, and indicators instead of counts
cv2 = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=10, max_df=0.7)

X_train_cv2 = cv2.fit_transform(X_train)
X_test_cv2  = cv2.transform(X_test)



#pd.DataFrame(X_train_cv2.toarray(), columns=cv2.get_feature_names()).head()


############################# LOGISTIC REGRESSION ############################

lr = LogisticRegression()
lr.fit(X_train_cv1, y_train)
y_pred_cv1 = lr.predict(X_test_cv1)



# Train the second model
lr.fit(X_train_cv2, y_train)
y_pred_cv2 = lr.predict(X_test_cv2)


def conf_matrix(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'], 
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu");

    true_neg, false_pos = cm[0]
    false_neg, true_pos = cm[1]

    accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg),3)
    precision = round((true_pos) / (true_pos + false_pos),3)
    recall = round((true_pos) / (true_pos + false_neg),3)
    f1 = round(2 * (precision * recall) / (precision + recall),3)

    cm_results = [accuracy, precision, recall, f1]
    return cm_results


cm1=conf_matrix(y_test, y_pred_cv1)
#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/cm1_lr1.svg')

cm2=conf_matrix(y_test, y_pred_cv2)
#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/cm2_lr2.svg')

# Compile all of the error metrics into a dataframe for comparison
results = pd.DataFrame(list(zip(cm1, cm2)))
results = results.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results.columns = ['LogReg1', 'LogReg2']
results
#            LogReg1  LogReg2
# Accuracy     0.842    0.919
# Precision    0.864    0.932
# Recall       0.914    0.952
# F1 Score     0.888    0.942


################################### NAIVE BAYES ####################################


# Fit the first Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X_train_cv1, y_train)

y_pred_cv1_nb = mnb.predict(X_test_cv1)


# Fit the second Naive Bayes model
from sklearn.naive_bayes import BernoulliNB

bnb = BernoulliNB()
bnb.fit(X_train_cv2, y_train)

y_pred_cv2_nb = bnb.predict(X_test_cv2)

# Here's the heat map for the first Naive Bayes model
cm3 = conf_matrix(y_test, y_pred_cv1_nb)
#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/cm3_nb1.svg')

# Here's the heat map for the second Naive Bayes model
cm4 = conf_matrix(y_test, y_pred_cv2_nb)
# plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/cm4_nb2.svg')

# Compile all of the error metrics into a dataframe for comparison
results_nb = pd.DataFrame(list(zip(cm3, cm4)))
results_nb = results_nb.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results_nb.columns = ['NB1', 'NB2']
results_nb

results = pd.concat([results, results_nb], axis=1)
results

#            LogReg1  LogReg2    NB1     NB2
#Accuracy     0.842    0.919     0.797   0.846
#Precision    0.864    0.932     0.857   0.895
#Recall       0.914    0.952     0.848   0.880
#F1 Score     0.888    0.942     0.852   0.887


################### TFID INSEAD OF COUNT VECTORISER#################

# Create TF-IDF versions of the Count Vectorizers created earlier in the exercise
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf1 = TfidfVectorizer(stop_words=stop, ngram_range=(1, 1), min_df=10, max_df=0.7)
X_train_tfidf1 = tfidf1.fit_transform(X_train)
X_test_tfidf1  = tfidf1.transform(X_test)
pd.DataFrame.sparse.from_spmatrix(X_train_tfidf1)
pd.DataFrame.sparse.from_spmatrix(X_test_tfidf1)
tfidf2 = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=10, max_df=0.7)
X_train_tfidf2 = tfidf2.fit_transform(X_train)
X_test_tfidf2  = tfidf2.transform(X_test)


# Fit the first logistic regression on the TF-IDF data
lr.fit(X_train_tfidf1, y_train)
y_pred_tfidf1_lr = lr.predict(X_test_tfidf1)
cm5 = conf_matrix(y_test, y_pred_tfidf1_lr)
#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/cm5_tf_idf_lr1.svg')


# Fit the second logistic regression on the TF-IDF data
lr.fit(X_train_tfidf2, y_train)
y_pred_tfidf2_lr = lr.predict(X_test_tfidf2)
cm6 = conf_matrix(y_test, y_pred_tfidf2_lr)
#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/cm6_tf_idf_lr2.svg')


# Fit the first Naive Bayes model on the TF-IDF data
mnb.fit(X_train_tfidf1.toarray(), y_train)
y_pred_tfidf1_nb = mnb.predict(X_test_tfidf1)
cm8 = conf_matrix(y_test, y_pred_tfidf1_nb)
#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/cm8_tf_idf_nb1.svg')


# # Fit the second Naive Bayes model on the TF-IDF data
bnb.fit(X_train_tfidf2.toarray(), y_train)
y_pred_tfidf2_nb = bnb.predict(X_test_tfidf2)
cm9 = conf_matrix(y_test, y_pred_tfidf2_nb)
#plt.savefig('/Users/jsong/Documents/durg-recommendation/fig/cm9_tf_idf_nb2.svg')

# Compile all of the error metrics into a dataframe for comparison
results_tf = pd.DataFrame(list(zip(cm5, cm6, cm8, cm9)))
results_tf = results_tf.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results_tf.columns = ['LR1-TFIDF', 'LR2-TFIDF', 'NB1-TFIDF', 'NB2-TFIDF']
results_tf

results = pd.concat([results, results_tf], axis=1)
results


#            LR1-TFIDF  LR2-TFIDF  NB1-TFIDF  NB2-TFIDF
# Accuracy       0.845      0.877      0.792      0.846
# Precision      0.862      0.884      0.782      0.895
# Recall         0.924      0.946      0.968      0.880
# F1 Score       0.892      0.914      0.865      0.887