#!/usr/bin/env python

from __future__ import print_function

from keras.engine import Layer
from keras import backend as K


class Interleave(Layer):
    """Special-purpose layer for interleaving sequences.

    Intended to merge a sequence of word vectors with a sequence of
    vectors representing dependencies a (word, dependency, word)
    pattern. Note that word vectors other than the first and the last
    are duplicated.

    For example, given

        [ [w11 w12 ...] [w21 w22 ...] [w31 w32 ...] ... ]
        [ [d11 d12 ...] [d21 d22 ...] ... ]

    produces

        [ [w11 w12 ... d11 d12 ... w21 w22 ...]
          [w21 w22 ... d21 d22 ... w31 w32 ...] ... ]

    (first dimension for batch not shown.)
    """
    
    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) != 2:
            raise ValueError('Interleave must be called with a list '
                             'of two tensors')
        if any(m for m in mask if m is not None):
            raise NotImplementedError('mask for Interleave')
        a, b = inputs
        if K.ndim(a) != 3 or K.ndim(b) != 3:
            raise ValueError('Interleaved tensors must have ndim 3')
        # Concatenate the sequences so that each item in b is preceded
        # by an item in a and followed by the next item in a.
        return K.concatenate([a[:, :-1, :], b, a[:, 1:, :]], axis=2)

    def get_output_shape_for(self, input_shape):
        a_shape, b_shape = input_shape
        return (a_shape[0], b_shape[1], 2*a_shape[2]+b_shape[2])


if __name__ == '__main__':
    # Example
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, Embedding, LSTM
    
    # 5-dim word embeddings, 4-dim dep embeddings
    we = np.arange(10).repeat(5).reshape((-1, 5))
    de = np.arange(10).repeat(4).reshape((-1, 4)) * 10

    # Inputs are sequences of three words and two dependencies
    w_in = Input(shape=(3,))
    d_in = Input(shape=(2,))
    # w_emb = Embedding(we.shape[0], we.shape[1], weights=[we],
    #                   name="L_Embedding_Word", mask_zero=True)(w_in)
    # d_emb = Embedding(de.shape[0], de.shape[1], weights=[de],
    #                   name="L_Embedding_DT", mask_zero=True)(d_in)
    w_emb = Embedding(we.shape[0], we.shape[1], weights=[we],
                      name="L_Embedding_Word")(w_in)
    d_emb = Embedding(de.shape[0], de.shape[1], weights=[de],
                      name="L_Embedding_DT")(d_in)
    lstm_words = LSTM(output_dim=2, activation=None,
                      name="L_LSTM_Word", return_sequences=True)(w_emb)
    lstm_dt = LSTM(output_dim=3, activation=None,
                   name="L_LSTM_DT", return_sequences=True)(d_emb)
    out = Interleave(name="OUT")([lstm_words, lstm_dt])
    model = Model(input=[w_in, d_in], output=out)
    model.compile('adam', 'mse')
    
    sentence1_words = np.asanyarray([[0,1,2]])
    sentence_1_deps = np.asanyarray([[1,0]])

    print(model.predict([sentence1_words, sentence_1_deps]))
