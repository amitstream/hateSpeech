#!/usr/local/bin/python3.9

import streamlit as st

import datetime
print(datetime.datetime.now(),"Program start.")

#import nltk
#from nltk.corpus import stopwords
import re
#import pandas as pd
#import nltk

import pickle
#import re
#import string
import numpy as np
#import pandas as pd

#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split

#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional


print(datetime.datetime.now(),"Finished import.")
st.text("Hate Speech Detector")
sentence=st.text_input('Sentence to analyze')

labels=['Homophobe', 'Sexist', 'OtherHate', 'NotHate', 'Religion', 'Racist']

# LIB LIB str_punc = string.punctuation.replace(',', '').replace("'",'')
def clean(text):
    global str_punc
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text 

tokenizer = Tokenizer()
# LIB LIB le = LabelEncoder()

print(datetime.datetime.now(),"Program. About to load the model.")
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

print(datetime.datetime.now(),"Program. Finished loading the model.")
print("*************\nSentence:",sentence)
sentence = clean(sentence)
sentence = tokenizer.texts_to_sequences([sentence])
sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
p=clf2.predict(sentence)
print("Prediction:",p)
a=np.argmax(p)
print("ArgMax:",a)
result=labels[a]
#result = le.inverse_transform(np.argmax(clf2.predict(sentence), axis=-1))[0]
proba =  np.max(clf2.predict(sentence))
print(f"{result} : {proba}\n\n")
st.text(f"{result}")
print(datetime.datetime.now(),"Program end.")
