#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 00:53:24 2020

@author: AkashTyagi
"""
import re
import string
import operator
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

lemm = WordNetLemmatizer()
# =============================================================================
# Loading Dataset
# =============================================================================
path = './dataset/'
df = pd.read_csv(path+'train.csv') 
test_df = pd.read_csv(path+'test.csv') 

# =============================================================================
# Loading Embeddings
# =============================================================================
import gensim
W2V_PATH  = "/home/qainfotech/embeddings/GoogleNews-vectors-negative300.bin.gz"
embeddings = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True, limit=100000)
#embeddings.init_sims(replace=True)

# =============================================================================
#  Data Preprocessing
# =============================================================================
print(df.columns)
print(df.head())
qid = df["qid"]
df.drop(['qid'], axis = 1, inplace = True) # Qid is not needed rightnow
print(df.head())

test_qid = df


print("Sincere questions: ",len(df[df['target']==0]))
print("InSincere questions: ",len(df[df['target']==1]))
print("Data is higly imbalanced, Our evaluation matrix needs to be good ")

df[df.target==None]
df[df.question_text ==None]
print(" NO missing Values ")

question_len = np.array(df.question_text.str.len())
unique, count = np.unique(question_len,return_counts=True)
print(np.asarray((unique, count)).T) # Question len range lies in 15-250

# removing question with len more than 250, less than 15
df.shape[0]
df.drop(index=df[df.question_text.str.len()<15].index,axis=0,inplace=True)
df.shape[0]
df.drop(index=df[df.question_text.str.len()>250].index,axis=0,inplace=True)
df.shape[0]


def build_vocab(word_list):
    vocab = collections.defaultdict(int)
    for sentence in tqdm(word_list,position=0, leave=True):
        for word in sentence:
            vocab[word]+=1
    return vocab


def vocab_in_embedding(vocab,embedding):
    found_in_vocab = {}
    vocab_count = 0
    not_found_in_vocab = {}
    non_vocab_count = 0
    
    for word in vocab:
        try:
            found_in_vocab[word] = embedding[word]
            vocab_count += vocab[word]
        except:
            try:
                found_in_vocab[word] = embedding[lemm.lemmatize(word)]
                vocab_count += vocab[word]
            except:
                not_found_in_vocab[word] = vocab[word]
                non_vocab_count += vocab[word]
            
    print('Found embeddings for {:.2%} of vocab'.format(len(found_in_vocab) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(vocab_count / (non_vocab_count + vocab_count)))
    sorted_x = sorted(not_found_in_vocab.items(), key=operator.itemgetter(1))[::-1]
    return sorted_x
    
    
word_list = [sentence.split() for sentence in df['question_text'].values]
vocab = build_vocab(word_list)
words_not_found_in_vocab  = vocab_in_embedding(vocab,embeddings)
print("Top words not found:\n",words_not_found_in_vocab[:20])
print("Words with question mark, some stop words and integers are not present in vocab.")

# Dealing with punctuations.
def remove_punc(sentence):
    for word in con:
        if word in sentence:
            sentence = re.sub(word,con[word].split("/")[0],sentence)
            
    for punc in string.punctuation:
        if punc in sentence:
            sentence = sentence.replace(punc," ")
    return sentence
    
word_list = [remove_punc(sentence).split() for sentence in tqdm(df['question_text'].values)]
vocab = build_vocab(word_list)
words_not_found_in_vocab  = vocab_in_embedding(vocab,embeddings)
print("Top words not found:\n",words_not_found_in_vocab[:20])
"4" in embeddings
"9" in embeddings
"10" in embeddings # embedding contains just 0-9

# Removing numbers
def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


# Final sentence clearning
def clean_text(sentence):
    for word in con:
        if word in sentence:
            sentence = re.sub(word,con[word].split("/")[0],sentence)
    for punc in string.punctuation:
        if punc in sentence:
            sentence = sentence.replace(punc," ")
    return clean_numbers(sentence)

df.reset_index(drop=True,inplace=True)


for i in tqdm(range(len(df['question_text'])),position=0,leave=True):    
    df.set_value(i,'question_text',clean_text(df.loc[i]['question_text']))

word_list = [sentence.split() for sentence in tqdm(df['question_text'].values)]    
vocab = build_vocab(word_list)
words_not_found_in_vocab  = vocab_in_embedding(vocab,embeddings)
print("Top words not found:\n",words_not_found_in_vocab[:10])

## split to train and val
train_df, val_df = train_test_split(df, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 250 # max number of words in a question to use

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_X = train_df["question_text"]
val_X = val_df["question_text"]
test_X = test_df["question_text"]

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X)) # Created a vocab of words
print("Vocab len: ",len(tokenizer.index_word))


train_X = tokenizer.texts_to_sequences(train_X) # Converting text into respective index loc
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)


## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen) 
print(train_X[0])
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen) #adds zero for paddings and convert into numpy array

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values

vocab_size = len(tokenizer.word_index)+1

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Flatten, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=maxlen))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.summary()

model.fit(x=train_X, y=train_y,batch_size=512, epochs=10, validation_data=(val_X, val_y))






con = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}



