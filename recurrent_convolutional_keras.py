# Michael A. Alcorn (malcorn@redhat.com)
# A (modified) implementation of the Recurrent Convolutional Neural Network (RCNN) found in [1].
# [1] Siwei, L., Xu, L., Kang, L., and Zhao, J. 2015. Recurrent convolutional
#         neural networks for text classification. In AAAI, pp. 2267-2273.
#         http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745

import gensim
import numpy as np
import random
import socket
import string
import sys
import time

from keras.layers import Dense, Input, Lambda, merge, LSTM, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.models import Model

word2vec = gensim.models.Word2Vec.load("word2vec.gensim")
# We add an additional row of zeros to the embeddings matrix to represent unseen words and the NULL token.
embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype = "float32")
embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

MAX_TOKENS = word2vec.syn0.shape[0]
embedding_dim = word2vec.syn0.shape[1]
hidden_dim_1 = 200
hidden_dim_2 = 100
NUM_CLASSES = 10

document = Input(shape = (None, ), dtype = "int32")
left_context = Input(shape = (None, ), dtype = "int32")
right_context = Input(shape = (None, ), dtype = "int32")

embedder = Embedding(MAX_TOKENS + 1, embedding_dim, weights = [embeddings], trainable = False)
doc_embedding = embedder(document)
l_embedding = embedder(left_context)
r_embedding = embedder(right_context)

# I use LSTM RNNs instead of vanilla RNNs as described in the paper.
forward = LSTM(hidden_dim_1, return_sequences = True)(l_embedding) # See equation (1).
backward = LSTM(hidden_dim_1, return_sequences = True, go_backwards = True)(r_embedding) # See equation (2).
together = merge([forward, doc_embedding, backward], mode = "concat", concat_axis = 2) # See equation (3).

semantic = TimeDistributed(Dense(hidden_dim_2, activation = "tanh"))(together) # See equation (4).

# Keras provides its own max-pooling layers, but they cannot handle variable length input
# (as far as I can tell). As a result, I define my own max-pooling layer here.
pool_rnn = Lambda(lambda x: x.max(axis = 1), output_shape = (hidden_dim_2, ))(semantic) # See equation (5).

output = Dense(NUM_CLASSES, input_dim = hidden_dim_2, activation = "sigmoid")(pool_rnn) # See equations (6) and (7).

model = Model(input = [document, left_context, right_context], output = output)
model.compile(optimizer = "adadelta", loss = "categorical_crossentropy", metrics = ["accuracy"])

doc_as_array = np.array([[1, 2, 3, 4]])
# We shift the document to the right to obtain the left-side contexts.
left_context_as_array = np.array([[MAX_TOKENS, 1, 2, 3]])
# We shift the document to the left to obtain the right-side contexts.
right_context_as_array = np.array([[2, 3, 4, MAX_TOKENS]])
target = np.array([NUM_CLASSES * [0]])
target(3) = 1

history = model.fit([doc_as_array, left_context_as_array, right_context_as_array], target, nb_epoch = 1, verbose = 0)
loss = history.history["loss"][0]
