import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM


filepath = tf.keras.utils.get_file('shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
characters = sorted(set(text))
char_to_index = dict((c,i) for i, c in enumerate(characters))
index_to_char = dict((i,c) for i, c in enumerate(characters))
seq_length = 40 
step_size = 3
sentences=[]
next_char =[]
for i in range (0, len(text)-seq_length, step_size):
    sentences.append(text[i: i + seq_length])
    next_char.append(text[i + seq_length])

x = np.zeros((len(sentences), seq_length, len(characters)), dtype = bool)
y = np.zeros((len(sentences),len(characters)), dtype = bool)

for i, staz in enumerate(sentences):
    for t, char in enumerate(staz):
        x[i,t,char_to_index[char]]=1
    y[i, char_to_index[next_char[i]]] =1

model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x,y,batch_sie=256, epochs=4)

def sample(preds,temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds= np.exp(preds)
    preds=exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length,temperature):
    start_index = random.randint(0, len(text)-seq_length-1)
    generated = ''
    sentence = text[start_index: start_index + seq_length]
    generated += sentence
    return generated

print(generate_text(300, 0.2))