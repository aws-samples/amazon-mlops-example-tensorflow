import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np
import os


src_bucket = os.getenv("BUCKET_NAME")

EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 40000

class attention(Layer):
    '''
    Defines an attention layer that uses a simple MLP to get attention scores
    '''
    def __init__(self, return_sequences=True,activation=None):
        self.return_sequences = return_sequences
        self.activation = tf.keras.activations.get(activation)
        super(attention,self).__init__()
        
    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        super(attention,self).build(input_shape)
        
    def call(self, x):
        e = self.activation(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a       
        if self.return_sequences:
            return output       
        return K.sum(output, axis=1)
        
class MultiLabelPrecision(tf.keras.metrics.Metric):
    '''
    Custom metric calculation - precision across multi labels
    '''
    def __init__(self, name='MultiLabelPrecision', **kwargs):       
        super(MultiLabelPrecision, self).__init__(name=name, **kwargs)
        self.tp = tf.Variable(0, dtype = 'float32')
        self.trues = tf.Variable(0, dtype = 'float32')

    def update_state(self, y_true, y_pred,sample_weight=None):
        ## converting to a 6 * None matrix
        corrects = tf.transpose(tf.cast(y_true,'float32'))
        preds = tf.transpose(tf.math.round(y_pred))
        ## converting to booleans
        booltrue = tf.equal(corrects,tf.constant(1.0))
        boolpred = tf.equal(preds,tf.constant(1.0))
        ## logical and to get true positives - including multi labels
        self.tp.assign_add(tf.reduce_sum(tf.cast(tf.math.logical_and(booltrue,boolpred),'float32')))
        ## sum to get all positives 
        self.trues.assign_add(tf.math.reduce_sum(corrects))
                      
    def result(self):     
        return self.tp/self.trues

    def reset_states(self):
        self.tp.assign(0)
        self.trues.assign(0)

def define_network(embedding_layer):
    '''
    Define LSTM network with an attention layer
    '''
    sequence_input = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    ## If pretrained embedding layer is not given, train your own
    if embedding_layer == "none":
        embedded_sequences = tf.keras.layers.Embedding(MAX_NB_WORDS,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH)(sequence_input)
    else:
        embedded_sequences = embedding_layer(sequence_input)
    lstm = Bidirectional(LSTM(100,dropout = 0.2, recurrent_dropout = 0.2,return_sequences=True))(embedded_sequences)
    lstm = LayerNormalization()(lstm)
    attentionlstm = attention(return_sequences=False,activation='tanh')(lstm)
    s = Dense(6,activation='sigmoid')(attentionlstm)
    model_LSTM = tf.keras.Model(inputs=[sequence_input],outputs=[s])
    print(model_LSTM.summary())
    return model_LSTM

