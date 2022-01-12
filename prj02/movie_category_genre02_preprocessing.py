import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
pd.set_option('display.unicode.east_asian_width', True)

df = pd.read_csv('./crawling/movie_genre_all.csv')
df.dropna(inplace=True)
print(df.head())
print(df.info())
X = df['summary']
Y = df['genre']

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
label = encoder.classes_
with open('./models/movie_encoder_2000.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)


stopwords = pd.read_csv('../../PRJ_Movie_for_you-1/crawling_data/stopwords.csv', index_col=0)
okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
    words = []
    for word in X[i]:
        if len(word) > 1:
            if word not in list(stopwords['stopword']):
                words.append(word)
    X[i] = ' '.join(words)
    print(X[i])


token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
with open('./models/movie_token_2000.pickle', 'wb') as f:
    pickle.dump(token, f)
print(tokened_X[:10])

wordsize = len(token.word_index) + 1
print(wordsize)

for i in range(len(tokened_X)):
    if 2000 < len(tokened_X[i]):
        tokened_X[i] = tokened_X[i][:2000]

X_pad = pad_sequences(tokened_X, 2000)
print(X_pad[:10])

X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./crawling/movie_data_max_{}_wordsize_{}'.format(2000, wordsize), xy)

