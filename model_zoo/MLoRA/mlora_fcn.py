import tensorflow as tf
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
from .domain_norm import DomainNorm
from tensorflow.python.keras import layers, backend, callbacks
from tensorflow.python.keras.layers import Layer, Dropout


class MLoRAFCN(Layer):

    def __init__(self,
                 n_domain,
                 units,

                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 lora_r=4,
                 lora_reduce=-1,
                 dropout_rate=0.5,
                 is_finetune=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(MLoRAFCN, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.n_domain = n_domain

        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        print("self.kernel_initializer: ", self.kernel_initializer)
        print("self.bias_initializer: ", self.bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout_rate = dropout_rate
        self.supports_masking = False
        self.lora_r = lora_r
        if lora_r < 1 and lora_reduce >= 1:
            self.lora_r = max(int(units/lora_reduce), 1)

        self.is_finetune = tf.constant(1.0 if is_finetune else 0.0, dtype=tf.float32)

    def build(self, input_shape):
        input_shape, domain_indicator_shape = input_shape
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = [InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value}),
                           InputSpec(shape=domain_indicator_shape)]

        # Domain
        self.a_kernel = self.add_weight(
            "A_Kernel",
            shape=[self.n_domain, input_shape[-1].value, self.lora_r],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)


        print("self.a_kernel", self.a_kernel)
        # self.a_kernel1 = self.is_finetune * self.a_kernel1 + (1.0-self.is_finetune) * tf.stop_gradient(self.a_kernel1)


        self.b_kernel = self.add_weight(
            "B_Kernel",
            shape=[self.n_domain, self.lora_r, self.units],
            initializer=self.bias_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        print("self.b_kernel", self.b_kernel)
        # self.b_kernel1 = self.is_finetune * self.b_kernel1 + (1.0-self.is_finetune) * tf.stop_gradient(self.b_kernel1)


        if self.use_bias:
            self.domain_bias = self.add_weight(
                "domain_bias",
                shape=[self.n_domain, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
            print("self.domain_bias", self.domain_bias)
            # self.domain_bias1 = self.is_finetune * self.domain_bias1 + (1.0 - self.is_finetune) * tf.stop_gradient(self.domain_bias1)
        else:
            self.domain_bias = None

        # MLP weight and bias
        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        print("self.kernel", self.kernel)
        # self.domain_bias = self.is_finetune * tf.stop_gradient(self.kernel) + (1.0 - self.is_finetune) * self.kernel

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
            print("self.bias", self.bias)
            # self.bias = self.is_finetune * tf.stop_gradient(self.bias) + (1.0 - self.is_finetune) * self.bias
        else:
            self.bias = None


        self.dropout_layers = Dropout(self.dropout_rate)

        self.built = True

    # def _domain_partition(self, idx):
    #     p_kernel = nn.embedding_lookup(self.PN_Kernel, idx)
    #     p_bias = nn.embedding_lookup(self.PN_Bias, idx)
    #
    #     self.kernel = tf.multiply(self.Shred_Kernel, p_kernel)
    #     if self.use_bias:
    #         self.bias = tf.add(self.Shared_Bias, p_bias)
    #     else:
    #         self.bias = None

    def call(self, inputs, training=None, **kwargs):
        # Unpack to original inputs and domain_indicator
        inputs, domain_indicator = inputs
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)

        #MLP
        outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        outputs = self.dropout_layers(outputs, training=training)

        # domain
        idx = tf.cast(domain_indicator[0, 0], tf.int32)
        domain_a_kernel = nn.embedding_lookup(self.a_kernel, idx)
        domain_b_kernel = nn.embedding_lookup(self.b_kernel, idx)
        if self.use_bias:
            domain_bias = nn.embedding_lookup(self.domain_bias, idx)
        domain_outputs = gen_math_ops.mat_mul(inputs, domain_a_kernel)
        domain_outputs = gen_math_ops.mat_mul(domain_outputs, domain_b_kernel)
        if self.use_bias:
            domain_outputs = nn.bias_add(domain_outputs, domain_bias)
        # domain_outputs1 = DomainNorm(self.n_domain_1)([domain_outputs1, domain_indicator1])
        outputs += domain_outputs


        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            "lora_r": self.lora_r,
            'n_domain': self.n_domain,
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MLoRAFCN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
