import nltk
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from gensim.models import word2vec
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing import sequence
from keras.optimizers import SGD
from sklearn import preprocessing

stops = set(stopwords.words("english"))

num_features = 300    # Word vector dimensionality
min_word_count = 1    # Minimum word count. If a word occurs atleast this number of times it is kept else it is removed.
num_workers = 4       # Number of threads to run in parallel
context = 5           # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

tree = ET.parse("/home/jeet/Academics/CS671/Project/Restaurants_Train.xml")
corpus = tree.getroot()
s = [] # List of list of word tokens.
sentences = corpus.findall('.//sentence')
for sent in sentences:
    tokens = nltk.word_tokenize(sent.find('text').text)
    s.append(tokens)

model = word2vec.Word2Vec(s, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "300features_1minwords_10context"
model.save("my_name_w2v_withstopwords")
# model = word2vec.Word2Vec()
model = word2vec.Word2Vec().load("my_name_w2v_withstopwords")
model = model.load("my_name_w2v_withstopwords")

le = preprocessing.LabelEncoder()
tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS",
        "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB", "PRP$", "WP$", ":", "SYM", "$", "--", ".", ",", "(", ")", "''",
        "#"]

le.fit(tags)

train_inp = []
train_out = []
i=0

sentences = corpus.findall('.//sentence')
for sent in sentences:
    w = []
    s = nltk.word_tokenize(sent.find('text').text)
    tags_for_sent = nltk.pos_tag(s)
    ind = [0] * len(s)
    i=0
    
    for t in s:
        ohe = [0]*45
        ohe[le.transform(tags_for_sent[i][1])] = 1
        w.append(np.concatenate([model[t], ohe]))
        i = i+1

    train_inp.append(w)
    aspectTerms = sent.find('aspectTerms')
    if len(aspectTerms):
        aspectTerm = aspectTerms.findall('aspectTerm')
        if (aspectTerm):
            for aspect_term in aspectTerm:
                try:
                    ind[s.index(aspect_term.attrib['term'])] = 1
                except:
                    continue
    train_out.append(ind)

for i in range(len(train_inp)):
    train_inp[i] = np.lib.pad(np.asarray(train_inp[i]), ((0, 80-len(train_inp[i])), (0,0)), 'constant', constant_values=(0))

for i in range(len(train_out)):
    train_out[i] = np.lib.pad(train_out[i], (0, 80-len(train_out[i])), 'constant', constant_values=(0))

train_inp = np.array(train_inp)
train_out = np.array(train_out)
train_inp = train_inp.reshape(3044, 80, 345).astype('float32')
train_out = train_out.reshape(3044, 80)


pickle.dump(train_inp, open("train_inp.pkl", "wb"))
train_inp = pickle.load(open("train_inp.pkl", "rb"))

pickle.dump(train_out, open("train_out.pkl", "wb"))
train_out = pickle.load(open("train_out.pkl", "rb"))

model = Sequential()

model.add(Convolution1D(100, 2, border_mode="same", input_shape=(80, 345)))
model.add(Activation("tanh"))
model.add(MaxPooling1D(pool_length=5))
model.add(Convolution1D(50, 3, border_mode="same"))
model.add(Activation("tanh"))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("tanh"))

model.add(Dense(80))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_inp, train_out, validation_split=0.1,
          batch_size=10,
          nb_epoch=30
          )


