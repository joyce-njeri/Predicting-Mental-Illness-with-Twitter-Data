from tkinter import FALSE
import streamlit as st
import pickle
import base64
import pandas as pd
from keras.models import model_from_json
import time

# text preprocessing
import re, string, unicodedata
import nltk
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

st.write(
    """
    # Welcome to the Mental Health Predictor App!
    This app predicts the likelihood score of a person on Twitter
    developing a mental illness through analyzing their tweets.
    """
)

# get sample tweet
st.write("To begin, please enter a user's tweet below")
sample_tweet = st.text_input('Enter Tweet:', max_chars=250, placeholder='Start typing ...')

# Prediction
ok = st.button('Make Prediction')

# Data upload
st.sidebar.markdown('## Import Dataset') 
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

score = FALSE
df = pd.DataFrame()
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f'<h3 "> Data Preview </h3>',unsafe_allow_html=True)
    pd.set_option('display.max_colwidth', None)
    st.write(df.head(10))

    score = st.checkbox('Check to predict scores')

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

# preprocess
def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    words = lemmatize_verbs(words)
    return words

def preprocess(sample):
    sample = remove_URL(sample)
    # Tokenize
    words = nltk.word_tokenize(sample)

    # Normalize
    words = normalize(words)
    
    temp = " ".join(word for word in words)
    return temp

# preprocess_loaded = data["preprocess"]
pad_sequences_loaded = data["pad_sequences"]
tokenizer_loaded = data["tokenizer"]

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

if ok:
    def predictor(sample_tweet):
        sample = preprocess(sample_tweet)
        sample = pad_sequences_loaded(tokenizer_loaded.texts_to_sequences(sample), maxlen = 30)
        score = loaded_model.predict(sample)
        return "{0:.0%}".format(1-score.mean())

    start_time = time.time()
    pred = predictor(sample_tweet)
    print("--- %s seconds ---" % (time.time() - start_time))
    st.subheader(f"The likelihood of the user developing or having a mental illness reoccur is {pred}")

download = False
if (score == True) & (uploaded_file is not None):
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(f'<h3 "> Scores Overview </h3>',unsafe_allow_html=True)
    
    start_time = time.time()
    scores = []
    for i in range(0,df.shape[0]):
        clean = preprocess(df.tweet[i])
        sample = pad_sequences_loaded(tokenizer_loaded.texts_to_sequences(clean), maxlen = 30)
        y_pred = loaded_model.predict(sample)
        scores.append("{0:.0%}".format(1-y_pred.mean()))
    print("--- %s seconds ---" % (time.time() - start_time))
    
    df['scores'] = scores
    st.write(df.head(10))

    download = st.checkbox('Check to download scores')

## Data Export
def convert_df(df):
   return df.to_csv().encode('utf-8')

csv = convert_df(df)

if (download == True) & (score == True) & (uploaded_file is not None):
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('### Downloads')
    st.download_button(
    "Press to Download",
    csv,
    "scores.csv",
    "text/csv",
    key='download-csv'
    )

