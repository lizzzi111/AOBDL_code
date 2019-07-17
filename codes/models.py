from keras.layers import Embedding, SpatialDropout1D
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, Input
from keras.optimizers import RMSprop
import keras.backend as K
from keras.layers import Dense, Input, GRU, LSTM, Bidirectional, Dropout, CuDNNLSTM, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPool1D
from keras.engine.topology import Layer, InputSpec

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
      
      
      
class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def model_(model_type = 'BGRU_avg', **kwargs):
  
  if K.backend == 'tensorflow':        
        K.clear_session()
  
  input_layer     = Input(shape=(kwargs[maxlen],))
  embedding_layer = Embedding(kwargs[max_features], output_dim=kwargs[embed_dim], trainable=True)(input_layer)
  x               = SpatialDropout1D(kwargs[dropout_rate])(embedding_layer)
    
  
  if model_type == 'GRU':
    x = CuDNNGRU(units=kwargs[rec_units], return_sequences=True)(x)
  elif model_type == 'BGRU':
    x = Bidirectional(CuDNNGRU(units=kwargs[rec_units], return_sequences=True))(x)
  elif model_type == 'BGRU_avg':
    x = Bidirectional(CuDNNGRU(units=kwargs[rec_units], return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
  elif model_type == 'BGRU_att':
    x = Bidirectional(CuDNNGRU(units=kwargs[rec_units], return_sequences=True))(x)
    x = AttentionWithContext()(x)
  elif model_type == 'BGRU_max':
    x = Bidirectional(CuDNNGRU(units=kwargs[rec_units], return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
  elif model_type == 'LSTM':
    x = CuDNNLSTM(units=kwargs[rec_units], return_sequences=True)(x)
  elif model_type == 'BLSTM':
    x = Bidirectional(CuDNNLSTM(units=kwargs[rec_units], return_sequences=True))(x)
  elif model_type == 'CNN':
    x = Conv1D(kwargs[num_filters], 7, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(kwargs[num_filters], 7, activation='relu', padding='same')(x)
    x = GlobalMaxPooling1D()(x)
    
  output_layer = Dense(1, activation="sigmoid")(x)
  model = Model(inputs=input_layer, outputs=output_layer)
  model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['acc'])
    #print( model.summary())
  return model