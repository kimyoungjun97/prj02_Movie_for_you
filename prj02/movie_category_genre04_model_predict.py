import pandas as pd
import numpy as np

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

pd.set_option('display.unicode.east_asian_width', True)
# data load
df = pd.read_csv('./crawling/movie_new.csv')
X = df['summary']
Y = df['genre']

# target labeling
with open('./models/movie_encoder_2000.pickle', 'rb') as f:
    encoder = pickle.load(f)
labeled_Y = encoder.transform(Y)
label = encoder.classes_
onehot_Y = to_categorical(labeled_Y)
print(labeled_Y[:5])
print(label)
print(onehot_Y)

# 형태소 분리, 한 글자/불용어 제거
okt = Okt()
stopwords = pd.read_csv('../../PRJ_Movie_for_you-1/crawling_data/stopwords.csv', index_col=0)
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
    words = []
    for word in X[i]:
        if len(word) > 1:
            if word not in list(stopwords['stopword']):
                words.append(word)
    X[i] = ' '.join(words)
print(X[:5])


# titles tokenizing
with open('./models/movie_token_500.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

for i in range(len(tokened_X)):
    if 500 < len(tokened_X[i]):
        tokened_X[i] = tokened_X[i][:500]

# padding
X_pad = pad_sequences(tokened_X, 500)
print(X_pad[:5])


model = load_model('./models/movie_genre_classification_model.h5')
pred = model.predict(X_pad)
sample = 6
print('pred is ', pred[sample])
print('actual is ', onehot_Y[sample])
print('Target :', label[np.argmax(onehot_Y[sample])])
pred = pred[sample].tolist()
acc = 0
ans = []
while acc < 0.99:
    ans.append(label[pred.index(max(pred))])
    acc = acc + max(pred)
    pred[pred.index(max(pred))] = 0
print('Prediction after learning is ', ans)
