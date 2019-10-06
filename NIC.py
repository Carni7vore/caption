'''
File to define the model structure of NIC, based on the paper:

https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf

model: Define the NIC training model

greedy_inference_model: Define the model used in greedy search.
                        Please initialize it with trained NIC model by load_weights()

image_dense_lstm: Define the encoding part of model used in beam search
                  Please initialize it with trained NIC model by load_weights()

text_emb_lstm: Define the decoding part of model used in beam search
               Please initialize it with trained NIC model by load_weights()
'''

import numpy as np
from keras import backend as K
from keras import regularizers
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Embedding,
                          Input, Lambda, TimeDistributed, Concatenate)
from keras.models import Model
from inception_v3 import InceptionV3
units = 512
img_embedding_size= 511
def model(vocab_size, max_len, reg):

    # Image embedding
    base_model = InceptionV3(include_top=True, weights=None)
    weights_path = 'data/image_net.h5'
    # image_model = tf.keras.applications.InceptionV3(include_top=False,
    #                                                 weights='imagenet')
    base_model.load_weights(weights_path)
    # print(base_model.output_shape)
    for layer in base_model.layers[:312]:
        layer.trainable = False

    new_input = base_model.layers[0].input
    hidden_layer = base_model.get_layer('avg_pool').output

    image_model = Model(new_input, hidden_layer)

    image_input = Input(shape=(299,299,3))
    X_img= image_model(image_input)
    X_img = Dropout(0.5)(X_img)
    X_img = Dense(img_embedding_size, use_bias = False,
                        kernel_regularizer=regularizers.l2(reg),
                        name = 'dense_img')(X_img)
    senti_input = Input(shape=(1,))
    X_img = Concatenate(axis=-1)([senti_input, X_img])
    X_img = BatchNormalization(name='batch_normalization_img')(X_img)
    X_img = Lambda(lambda x : K.expand_dims(x, axis=1))(X_img)

    # Text embedding
    seq_input = Input(shape=(max_len,))
    X_text = Embedding(vocab_size, units, mask_zero = True, name = 'emb_text')(seq_input)
    X_text = Dropout(0.5)(X_text)

    # Initial States
    a0 = Input(shape=(units,))
    c0 = Input(shape=(units,))

    LSTMLayer = LSTM(units, return_sequences = True, return_state = True, dropout=0.5, name = 'lstm')

    # Take image embedding as the first input to LSTM
    _, a, c = LSTMLayer(X_img, initial_state=[a0, c0])

    A, _, _ = LSTMLayer(X_text, initial_state=[a, c])
    output = TimeDistributed(Dense(vocab_size, activation='softmax',
                                     kernel_regularizer = regularizers.l2(reg), 
                                     bias_regularizer = regularizers.l2(reg)), name = 'time_distributed_softmax')(A)

    return Model(inputs=[image_input,seq_input,senti_input, a0, c0], outputs=output, name='NIC')


def greedy_inference_model(vocab_size, max_len):
    base_model = InceptionV3(include_top=True, weights=None)
    weights_path = 'data/image_net.h5'
    # image_model = tf.keras.applications.InceptionV3(include_top=False,
    #                                                 weights='imagenet')
    base_model.load_weights(weights_path)
    # print(base_model.output_shape)
    for layer in base_model.layers[:312]:
        layer.trainable = False

    new_input = base_model.layers[0].input
    hidden_layer = base_model.get_layer('avg_pool').output

    image_model = Model(new_input, hidden_layer)

    
    EncoderDense = Dense(img_embedding_size, use_bias=False, name = 'dense_img')
    EmbeddingLayer = Embedding(vocab_size, units, mask_zero = True, name = 'emb_text')
    LSTMLayer = LSTM(units, return_state = True, name = 'lstm')
    SoftmaxLayer = Dense(vocab_size, activation='softmax', name = 'time_distributed_softmax')
    BatchNormLayer = BatchNormalization(name='batch_normalization_img')

    # Image embedding
    image_input = Input(shape=(299,299,3))
    X_img = image_model(image_input)
    X_img = EncoderDense(X_img)
    senti_input = Input(shape=(1,))
    X_img = Concatenate(axis=-1)([senti_input, X_img])
    X_img = BatchNormLayer(X_img)
    X_img = Lambda(lambda x : K.expand_dims(x, axis=1))(X_img)

    # Text embedding
    seq_input = Input(shape=(1,))
    X_text = EmbeddingLayer(seq_input)

    # Initial States
    a0 = Input(shape=(units,))
    c0 = Input(shape=(units,))

    a, _, c = LSTMLayer(X_img, initial_state=[a0, c0])

    x = X_text

    outputs = []
    for i in range(max_len):
        
        a, _, c = LSTMLayer(x, initial_state=[a, c])
        output = SoftmaxLayer(a)
        outputs.append(output)
        x = Lambda(lambda x : K.expand_dims(K.argmax(x)))(output)
        x = EmbeddingLayer(x)

    return Model(inputs=[image_input,seq_input,senti_input, a0, c0], outputs=outputs, name='NIC_greedy_inference_v2')


def image_dense_lstm():
    base_model = InceptionV3(include_top=True, weights=None)
    weights_path = 'data/image_net.h5'
    # image_model = tf.keras.applications.InceptionV3(include_top=False,
    #                                                 weights='imagenet')
    base_model.load_weights(weights_path)
    # print(base_model.output_shape)
    for layer in base_model.layers[:312]:
        layer.trainable = False

    new_input = base_model.layers[0].input
    hidden_layer = base_model.get_layer('avg_pool').output

    image_model = Model(new_input, hidden_layer)



    EncoderDense = Dense(units, use_bias = False, name = 'dense_img')
    BatchNormLayer = BatchNormalization(name = 'batch_normalization_img')
    LSTMLayer = LSTM(units, return_state = True, name = 'lstm')

    inputs = Input(shape=(299,299,3))
    X_img = image_model(inputs)
    X_img = EncoderDense(X_img)
    X_img = BatchNormLayer(X_img)
    X_img = Lambda(lambda x : K.expand_dims(x, axis=1))(X_img)

    a0 = Input(shape=(units,))
    c0 = Input(shape=(units,))

    a, _, c = LSTMLayer(X_img, initial_state=[a0, c0])

    return Model(inputs=[inputs, a0, c0], outputs=[a, c])


def text_emb_lstm(vocab_size):

    EmbeddingLayer = Embedding(vocab_size, units, mask_zero = True, name='emb_text')
    LSTMLayer = LSTM(units, return_state = True, name='lstm')
    SoftmaxLayer = Dense(vocab_size, activation='softmax', name='time_distributed_softmax')

    a0 = Input(shape=(units,))
    c0 = Input(shape=(units,))
    cur_word = Input(shape=(1,))

    X_text = EmbeddingLayer(cur_word)
    a, _, c = LSTMLayer(X_text, initial_state=[a0, c0])
    output = SoftmaxLayer(a)

    return Model(inputs=[a0, cur_word, c0], outputs=[output, a, c])
