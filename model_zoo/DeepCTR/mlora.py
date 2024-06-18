from tensorflow.python.keras.layers import Layer
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2

from deepctr.layers.activation import activation_layer
from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.keras import layers, backend, callbacks
from tensorflow.python.keras.layers import Layer, Dropout


class Mlora(Layer):


    def __init__(self, n_domain,
                 hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024,
                 lora_r=None,
                 lora_reduce=-1,
                 **kwargs
                 ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed
        self.n_domain = n_domain
        self.dropout_rate = dropout_rate

        if lora_reduce >= 1:
            self.lora_r = [max(int(self.hidden_units[i]/lora_reduce), 1) for i in range(len(self.hidden_units))]

        super(Mlora, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size, domain_indicator_shape = input_shape
        input_size = input_size[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]

        self.A_kernel = [self.add_weight(name='bias' + str(i),
                                     shape=(self.n_domain, hidden_units[i], self.lora_r[i]),
                                     initializer=glorot_normal(
                                             seed=self.seed),
                                     trainable=True) for i in range(len(self.hidden_units))]
        self.B_kernel = [self.add_weight(name='bias' + str(i),
                                     shape=(self.n_domain, self.lora_r[i], hidden_units[i+1]),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        self.lora_bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.n_domain, self.hidden_units[i]),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(Mlora, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        inputs, domain_indicator = inputs
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        assert rank<=2, 'error,rank > 2'

        for i in range(len(self.hidden_units)):
            #
            # fc = tf.nn.bias_add(tf.tensordot(
            #     deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            idx = tf.cast(domain_indicator[0, 0], tf.int32)
            domain_a_kernel = nn.embedding_lookup(self.A_kernel[i], idx)
            domain_b_kernel = nn.embedding_lookup(self.B_kernel[i], idx)
            domain_bias = nn.embedding_lookup(self.lora_bias[i], idx)

            # DNN
            dnn_fc = gen_math_ops.mat_mul(inputs, self.kernels[i])
            dnn_fc = nn.bias_add(dnn_fc, self.bias[i])
            dnn_fc = self.dropout_layers[i](dnn_fc, training=training)

            # domain

            lora_fc = gen_math_ops.mat_mul(inputs, domain_a_kernel)
            lora_fc= gen_math_ops.mat_mul(lora_fc, domain_b_kernel)
            lora_fc = nn.bias_add(lora_fc, domain_bias)

            fc = dnn_fc + lora_fc
            # fc = dnn_fc

            # if self.use_bn:
            #     fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            inputs = fc

        return inputs

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed, "lora_r":self.lora_r,'n_domain': self.n_domain}
        base_config = super(Mlora, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))