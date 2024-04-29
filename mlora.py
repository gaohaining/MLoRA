import os
import os.path as osp
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import train
from tensorflow.python.keras import layers, backend, callbacks
from tensorflow.python.keras.models import Model
from utils.auc import AUC  # This can be replaced by tf.keras.AUC when tf version >=1.12

from .auxiliary_net import AuxiliaryNet
from ..base_model import BaseModel
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
from tensorflow.python.keras import layers, backend, callbacks
from tensorflow.python.keras.layers import Layer, Dropout



class MLoRA(BaseModel):
    def __init__(self, dataset, config):
        super(MLoRA, self).__init__(dataset, config)

    def build_model(self):
        model = self.build_model_structure()
        model.summary()
        # Optimization
        if self.train_config['optimizer'] == 'adam':
            opt = train.AdamOptimizer(learning_rate=self.train_config['learning_rate'])
        else:
            opt = self.train_config['optimizer']

        model.compile(loss=self.train_config['loss'], optimizer=opt,
                      metrics=[AUC(num_thresholds=500, name="AUC")])
        return model

    def train(self):
        if 'freeze' in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == "True":
                backend.get_session().run(tf.global_variables_initializer())
        else:
            backend.get_session().run(tf.global_variables_initializer())

        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.dirname(self.checkpoint_path),
                                                     histogram_freq=self.train_config['histogram_freq'],
                                                     write_grads=True)
        train_sequence = list(range(self.n_domain))
        lock = False
        print("=" * 50, "train_from_ckp start", "=" * 50)
        print("train_from_ckp: ", self.train_config['train_from_ckp'])
        if self.train_config['train_from_ckp']:
            dir_file = osp.join(self.train_config['checkpoint_path'], self.model_config['name'],
                                            self.dataset.conf['name'], self.dataset.conf['domain_split_path'])
            ckps = os.listdir(dir_file)
            ckps.sort()
            self.checkpoint_path = osp.join(self.train_config['checkpoint_path'], self.model_config['name'],
                                            self.dataset.conf['name'], self.dataset.conf['domain_split_path'],
                                            ckps[-1],
                                            "model_parameters.h5")
            print("self.checkpoint_path: ", self.checkpoint_path)
            self.load_model(self.checkpoint_path)  # Load best model weights
        print("="*50, "train_from_ckp end", "="*50)

        if 'freeze' in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == "False":
                avg_loss, avg_auc, domain_loss, domain_auc = self.val_and_test("val")
                self.early_stop_step(avg_auc)

        for epoch in range(self.train_config['epoch']):
            print("Epoch: {}".format(epoch), "-" * 30)
            # Train
            random.shuffle(train_sequence)
            for idx in train_sequence:
                d = self.dataset.train_dataset[idx]
                # print("Train on: Domain_{}_{}".format(self.dataset.domain_1[idx], self.dataset.domain_2[idx]))
                old_time = time.time()
                self.model.fit(d['data'], steps_per_epoch=d['n_step'], verbose=2, callbacks=[],
                               epochs=epoch + 1, initial_epoch=epoch)
                print("Training time: ", time.time() - old_time)
            # Val
            print("Val Result: ")
            # avg_loss, avg_auc = self.val_and_test_total("val")
            avg_loss, avg_auc, domain_loss, domain_auc = self.val_and_test("val")
            # Early Stopping
            self.epoch = epoch
            if self.early_stop_step(avg_auc):
                break
            # Test
            print("Test Result: ")
            # test_avg_loss, test_avg_auc = self.val_and_test_total("test")
            test_avg_loss, test_avg_auc, domain_loss, domain_auc = self.val_and_test("test")
            # Lock the graph for better performance
            if not lock:
                graph = tf.get_default_graph()
                graph.finalize()
                lock = True

    def build_model_structure(self):
        inputs, x, domain_emb = self.build_inputs()

        domain_indicator = inputs[-1]


        # DomainLoRA
        if self.model_config['dense'] == "dense":
            for h_dim in self.model_config['hidden_dim']:
                x = layers.Dense(h_dim, activation="relu")(x)
        elif self.model_config['dense'] == "mlora":
            lora_r = self.model_config["lora_r"]
            lora_reduce = self.model_config["lora_reduce"]
            dropout_rate = self.model_config["dropout"]
            for h_dim in self.model_config['hidden_dim']:
                x = MLoRAFCN(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce, dropout_rate=dropout_rate,
                             is_finetune=self.train_config['train_from_ckp']
                             )([x, domain_indicator])
        elif "mlora_freeze" in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == 'True':
                lora_r = self.model_config["lora_r"]
                lora_reduce = self.model_config["lora_reduce"]
                dropout_rate = self.model_config["dropout"]
                for h_dim in self.model_config['hidden_dim']:
                    x = MLoRAFCN_PRE(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce, dropout_rate=dropout_rate,
                                 is_finetune=self.train_config['train_from_ckp'],
                                 )([x, domain_indicator])
            elif self.model_config['pretrain_judge'] == "False":
                lora_r = self.model_config["lora_r"]
                lora_reduce = self.model_config["lora_reduce"]
                dropout_rate = self.model_config["dropout"]
                for h_dim in self.model_config['hidden_dim']:
                    x = MLoRAFCN_FREEZE(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce, dropout_rate=dropout_rate,
                                 is_finetune=self.train_config['train_from_ckp'],
                                 )([x, domain_indicator])

        elif "mlora_star_freeze" in self.model_config['dense']:
            lora_r = self.model_config["lora_r"]
            lora_reduce = self.model_config["lora_reduce"]
            dropout_rate = self.model_config["dropout"]
            for h_dim in self.model_config['hidden_dim']:
                x = MLoRAFCN_FREEZE(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce,
                                    dropout_rate=dropout_rate,
                                    is_finetune=self.train_config['train_from_ckp'],
                                    )([x, domain_indicator])

        if "freeze" in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == 'False':
                output = layers.Dense(1, activation="sigmoid",trainable=False)(x)
            else:
                output = layers.Dense(1, activation="sigmoid")(x)
        else:
            output = layers.Dense(1, activation="sigmoid")(x)

        return Model(inputs=inputs, outputs=output)

    def build_inputs(self):
        user_input_layer = layers.Input(shape=(1,), dtype=tf.int32, name='uid')
        item_input_layer = layers.Input(shape=(1,), dtype=tf.int32, name='pid')

        domain_input_layer = layers.Input(shape=(1,), dtype=tf.int32, name='domain')

        user_emb = self.build_emb("user_emb", self.n_uid, self.model_config['user_dim'],
                                  trainable=self.train_config['emb_trainable'])(user_input_layer)  # B * 1 * d
        item_emb = self.build_emb("item_emb", self.n_pid, self.model_config['item_dim'],
                                  trainable=self.train_config['emb_trainable'])(item_input_layer)

        if "freeze" in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == 'False':
                domain_emb = layers.Embedding(self.n_domain, self.model_config['domain_dim'], name="domain_emb",
                                          trainable=False)(domain_input_layer)
            else:
                domain_emb = layers.Embedding(self.n_domain, self.model_config['domain_dim'], name="domain_emb")(
                    domain_input_layer
                )
        else:
            domain_emb = layers.Embedding(self.n_domain, self.model_config['domain_dim'], name="domain_emb")(
                domain_input_layer
            )

        input_feat = layers.Concatenate(-1)([user_emb, item_emb, domain_emb])  # B * 1 * 3d
        input_feat = layers.Flatten()(input_feat)  # B * 3d

        return [user_input_layer, item_input_layer, domain_input_layer], input_feat, layers.Flatten()(domain_emb)

    def build_emb(self, attr_name, n, dim, trainable=True):
        if self.train_config['load_pretrain_emb']:
            embedding_matrix = np.zeros((n, dim))
            emb_dict = getattr(self.dataset, attr_name)
            for key in sorted(emb_dict.keys()):
                embedding_vector = np.asarray(emb_dict[key].split(" "), dtype='float32')
                embedding_matrix[int(key)] = embedding_vector
            emb_layer = layers.Embedding(n, dim, name=attr_name,
                                         embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                         trainable=trainable)
        else:
            if 'freeze' in self.model_config['dense']:
                if self.model_config['pretrain_judge'] == 'False':
                    emb_layer = layers.Embedding(n, dim, name=attr_name,trainable=False)
                    return emb_layer
            emb_layer = layers.Embedding(n, dim, name=attr_name)
        return emb_layer
    def random_delete(self,train_sequence):

        a = random.randint(0,self.n_domain - 1)
        self.finetune_train_sequence = []
        self.finetune_train_sequence.append(train_sequence[a])
        train_sequence.pop(a)
        b = random.randint(0,self.n_domain - 2)
        self.finetune_train_sequence.append(train_sequence[b])
        train_sequence.pop(b)
        return train_sequence




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

        self.b_kernel = self.add_weight(
            "B_Kernel",
            shape=[self.n_domain, self.lora_r, self.units],
            initializer=self.bias_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        print("self.b_kernel", self.b_kernel)


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


        else:
            self.domain_bias = None

        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        print("self.kernel", self.kernel)

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
        else:
            self.bias = None


        self.dropout_layers = Dropout(self.dropout_rate)

        self.built = True


    def call(self, inputs, training=None, **kwargs):
        # Unpack to original inputs and domain_indicator
        inputs, domain_indicator = inputs
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)

        # Partition by domain
        idx = tf.cast(domain_indicator[0, 0], tf.int32)
        domain_a_kernel = nn.embedding_lookup(self.a_kernel, idx)
        domain_b_kernel = nn.embedding_lookup(self.b_kernel, idx)
        if self.use_bias:
            domain_bias = nn.embedding_lookup(self.domain_bias, idx)

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        outputs = self.dropout_layers(outputs, training=training)
        # outputs = layers.BatchNormalization()(outputs)

        # domain
        if rank > 2:
            # Broadcasting is required for the inputs.
            domain_outputs = standard_ops.tensordot(inputs, domain_a_kernel, [[rank - 1], [0]])
            domain_outputs = standard_ops.tensordot(domain_outputs, domain_b_kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                domain_outputs.set_shape(output_shape)
        else:
            domain_outputs = gen_math_ops.mat_mul(inputs, domain_a_kernel)
            domain_outputs = gen_math_ops.mat_mul(domain_outputs, domain_b_kernel)
        if self.use_bias:
            domain_outputs = nn.bias_add(domain_outputs, domain_bias)

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



class MLoRAFCN_PRE(Layer):

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
            trainable=False)
        print("self.a_kernel", self.a_kernel)


        self.b_kernel = self.add_weight(
            "B_Kernel",
            shape=[self.n_domain, self.lora_r, self.units],
            initializer=self.bias_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=False)
        print("self.b_kernel", self.b_kernel)


        if self.use_bias:
            self.domain_bias = self.add_weight(
                "domain_bias",
                shape=[self.n_domain, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=False)
            print("self.domain_bias", self.domain_bias)


        else:
            self.domain_bias = None

        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        print("self.kernel", self.kernel)

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
        else:
            self.bias = None

        self.dropout_layers = Dropout(self.dropout_rate)

        self.built = True

   

    def call(self, inputs, training=None, **kwargs):
        # Unpack to original inputs and domain_indicator
        inputs, domain_indicator = inputs
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)

        # Partition by domain
        idx = tf.cast(domain_indicator[0, 0], tf.int32)
        domain_a_kernel = nn.embedding_lookup(self.a_kernel, idx)
        domain_b_kernel = nn.embedding_lookup(self.b_kernel, idx)
        if self.use_bias:
            domain_bias = nn.embedding_lookup(self.domain_bias, idx)

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        outputs = self.dropout_layers(outputs, training=training)

        # domain
        if rank > 2:
            # Broadcasting is required for the inputs.
            domain_outputs = standard_ops.tensordot(inputs, domain_a_kernel, [[rank - 1], [0]])
            domain_outputs = standard_ops.tensordot(domain_outputs, domain_b_kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                domain_outputs.set_shape(output_shape)
        else:
            domain_outputs = gen_math_ops.mat_mul(inputs, domain_a_kernel)
            domain_outputs = gen_math_ops.mat_mul(domain_outputs, domain_b_kernel)
        if self.use_bias:
            domain_outputs = nn.bias_add(domain_outputs, domain_bias)

        outputs += domain_outputs
        if self.activation is not None:
            return self.activation(outputs)  
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



class MLoRAFCN_FREEZE(Layer):

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


        self.b_kernel = self.add_weight(
            "B_Kernel",
            shape=[self.n_domain, self.lora_r, self.units],
            initializer=self.bias_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        print("self.b_kernel", self.b_kernel)

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


        else:
            self.domain_bias = None

        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=False)
        print("self.kernel", self.kernel)

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=False)
            print("self.bias", self.bias)
        else:
            self.bias = None

        self.dropout_layers = Dropout(self.dropout_rate)

        self.built = True


    def call(self, inputs, training=None, **kwargs):
        # Unpack to original inputs and domain_indicator
        inputs, domain_indicator = inputs
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)

        # Partition by domain
        idx = tf.cast(domain_indicator[0, 0], tf.int32)
        domain_a_kernel = nn.embedding_lookup(self.a_kernel, idx)
        domain_b_kernel = nn.embedding_lookup(self.b_kernel, idx)
        if self.use_bias:
            domain_bias = nn.embedding_lookup(self.domain_bias, idx)

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        outputs = self.dropout_layers(outputs, training=training)

        # domain
        if rank > 2:
            # Broadcasting is required for the inputs.
            domain_outputs = standard_ops.tensordot(inputs, domain_a_kernel, [[rank - 1], [0]])
            domain_outputs = standard_ops.tensordot(domain_outputs, domain_b_kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                domain_outputs.set_shape(output_shape)
        else:
            domain_outputs = gen_math_ops.mat_mul(inputs, domain_a_kernel)
            domain_outputs = gen_math_ops.mat_mul(domain_outputs, domain_b_kernel)
        if self.use_bias:
            domain_outputs = nn.bias_add(domain_outputs, domain_bias)

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

