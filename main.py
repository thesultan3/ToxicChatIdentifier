import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
import spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("C:/Users/Ammaar/PycharmProjects/ToxicChatIdentifier/train.csv")
# print(data.head())
# print(data.shape)
# print(data.info)
# print(data.toxic.value_counts())


nlp = spacy.load('en_core_web_sm')
#using sm for early dev change to en_core_web_lg for increased accuracy near end

#custom stopword list after looking trying to remove erroneous highly rated words
stop_words = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves

explanation edits username hardcore metallica fan reverted vandalisms
closure gas voted new york dolls fac remove tamplate talk page
-pron- retired now.89.205.38.27 d'aww matches background colour
seemingly stuck thanks 21:51 january 11 2016 utc hey man trying
edit war 's guy talking instead care info real suggestions improvement
wondered section statistics need tidying exact format ie date etc -i
eg good_article_nominations#transport matt article(wow 93.161.107.169
dulithglow wonju.jpg
""".split()
)

#create punctuation list
punctuations = string.punctuation

parser = English()

#tokenizer function
def spacy_tokenizer(sentence):
    #create token object
    mytokens = parser(sentence)

    #lemmatize&lower case
    mytokens = [word.lemma_.lower().strip()
                for word in mytokens]

    # removing stop words
    mytokens = [word for word in mytokens if word not in stop_words
                and word not in punctuations]

    #return preprocessed tokens (for vectorization)
    return mytokens


#Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

#Basic function to clean text
def clean_text(text):
    #Removing spaces and converting text into lowercase
    return text.strip().lower()

#the vectorizer used to classify the words
bow_vector = CountVectorizer(tokenizer= spacy_tokenizer, ngram_range=(1,1))

"""
Used to check the most common words
docs=data['comment_text'].tolist()
word_count_vector=bow_vector.fit_transform(docs)
print(list(bow_vector.vocabulary_.keys())[:300])

tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)
"""

X = data['comment_text'] #the features we want to analyze (the text - independent var)
ylabels = data['toxic'] #the labels we want to test against (the answer - dependent var)

#splitting the dataset into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

#after testing with models and penalty this was the best configuration
classifier = LogisticRegression(max_iter=25000,solver='lbfgs')

#create BoW pipeline
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train, y_train)

from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Resutls
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision", metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall", metrics.recall_score(y_test, predicted))


#Presenting the results in confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predicted)
print(cm)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


