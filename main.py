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


#stopword list
nlp = spacy.load('en_core_web_sm')
#using sm for early dev change to en_core_web_lg for increased accuracy near end
stop_words = spacy.lang.en.stop_words.STOP_WORDS
#create punctuation list
punctuations = string.punctuation

parser = English()

#tokenizer function
def spacy_tokenizer(sentence):
    #create token object
    mytokens = parser(sentence)

    #lemmatize&lower case
    mytokens = [word.lemma_.lower().strip()
                if word.lemma_ != "-PRON-" else word.lower_
                for word in mytokens]

    # removing stop words
    mytokens = [word for word in mytokens if word not in stop_words
                and word not in punctuations]
    #make punctuations custom - allow exclamation marks they could be a good indicator

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

bow_vector = CountVectorizer(tokenizer= spacy_tokenizer, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)

X = data['comment_text'] #the features we want to analyze (the text - independent var)
ylabels = data['toxic'] #the labels we want to test against (the answer - dependent var)

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

classifier = LogisticRegression(solver='liblinear',max_iter=500)#was 1000 (doing 25x more iterations)

#create BoW pipeline
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train, y_train)

from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision", metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall", metrics.recall_score(y_test, predicted))

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