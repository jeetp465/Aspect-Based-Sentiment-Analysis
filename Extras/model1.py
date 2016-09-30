import xml.etree.ElementTree as ET
from gensim.models import word2vec
import nltk
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing import sequence
import numpy as np
from keras.optimizers import SGD

stops = set(stopwords.words("english"))
# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

tree = ET.parse("/home/rajat/Desktop/Restaurants_Train.xml")
corpus = tree.getroot()
s = []
sentences = corpus.findall('.//sentence')
for sent in sentences:
    tokens = nltk.word_tokenize(sent.find('text').text)
    #words = [w for w in tokens if not w in stops]
    s.append(tokens)

model = word2vec.Word2Vec(s, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save("my_name_w2v_withstopwords")
model = model.load("my_name_w2v_withstopwords")
#Laptop_term = []
# output the children nodes of root
#print (root)
tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB"]
train_inp = []
train_out = []
i=0
sentences = corpus.findall('.//sentence')
for sent in sentences:
    #print (sent.find('text').text)
    w = []
    s = nltk.word_tokenize(sent.find('text').text)
    tags = nltk.pos_tag(s)
    ind = [0] * len(s)
    for t in s:
        w.append(model[t])
    train_inp.append(w)
    aspectTerms = sent.find('aspectTerms')
    if (aspectTerms):
        aspectTerm = aspectTerms.findall('aspectTerm')
        if (aspectTerm):
            for aspect_term in aspectTerm:
                try:
                    ind[s.index(aspect_term.attrib['term'])] = 1
                except:
                    continue
    #print (aspect_term.attrib['term'])

    #aspectCategories = sent.find('aspectCategories')
    #aspectCategory = aspectCategories.findall('aspectCategory')
    #for aspectcateg in aspectCategory:
     #   print (aspectcateg.attrib['category'])
    train_out.append(ind)

#np.save('train_inp', train_inp)
#np.save('train_out', train_out)
#train_inp = np.load('train_inp.npy')
#train_out = np.load('train_out.npy')

for i in range(len(train_inp)):
    train_inp[i] = np.lib.pad(np.asarray(train_inp[i]), ((0, 80-len(train_inp[i])), (0,0)), 'constant', constant_values=(0))
    print(i)

for i in range(len(train_out)):
    train_out[i] = np.lib.pad(train_out[i], (0, 80-len(train_out[i])), 'constant', constant_values=(0))

train_inp = np.array(train_inp)
train_out = np.array(train_out)
train_inp = train_inp.reshape(3044, 80, 300).astype('float32')
train_out = train_out.reshape(3044, 80)
#train_inp = sequence.pad_sequences(train_inp, max_length=(65, 300))
#train_out = sequence.pad_sequences(train_out, max_length=65)
model = Sequential()

model.add(LSTM(300, input_shape=(80, 300)))
# model.add(Dense(80, 1, activation='softmax'))
model.add(Dense(80))
model.add(Activation('softmax'))
# model.add(Convolution2D(80, 3, 3, border_mode="same",
# 			input_shape=(1, 80, 300)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(80, 3, 3,  border_mode="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(500))
# model.add(Activation("relu"))
# # softmax classifier
# model.add(Dense(80))
# model.add(Activation("softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_inp, train_out, validation_split=0.1,
          batch_size=50,
          nb_epoch=10
          )

#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
