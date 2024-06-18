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
from .partitioned_norm import PartitionedNorm
from .domain_norm import DomainNorm
from .multi_domain_norm import MultiDomainNorm
from .multi_domain_norm2 import MultiDomainNorm as MultiDomainNorm2

# 正常LoRA
from .mlora_fcn import MLoRAFCN
# 跃层LoRA
from .mlora_fcn_skip import MLoRAFCN as MLoRAFCN_SKIP
# 先单独训练MLP
from .mlora_fcn_pretrain import MLoRAFCN as MLoRAFCN_PRE
# freeze W,训练lora
from .mlora_fcn_freeze import MLoRAFCN as MLoRAFCN_FREEZE
# with generator weight,pretrain and finetune
from .mlora_fcn_generator_pretrain import MLoRAFCN as MLoRAFCN_GENERATOR_PRETRAIN
from .mlora_fcn_generate import MLoRAFCN as MLoRAFCN_GENERATOR

from .auxiliary_net import AuxiliaryNet
from ..base_model import BaseModel


class MLoRA(BaseModel):
    def __init__(self, dataset, config):
        super(MLoRA, self).__init__(dataset, config)

    def build_model(self):
        # self.n_domain_1 = self.dataset.n_domain_1
        # self.n_domain_2 = self.dataset.n_domain_2
        # self.n_domain = self.dataset.n_domain
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
        # backend.get_session().run(tf.global_variables_initializer())

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


        # 80% themes to train,20% themes to finetune
        if "finetune" in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == 'True':
                train_sequence = self.random_delete(train_sequence)
                print('Start Using 80% themes to train Model,train_sequence is :',train_sequence)
            else:
                train_sequence = self.finetune_train_sequence
                print('Start Using 20% themes to finetune Model,finetune_sequence is :',train_sequence)

        # 保留pretrain的best auc
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
        # domain_indicator1 = inputs[-2]
        # domain_indicator2 = inputs[-1]
        # PartitionedNorm

        # freeze PareitionNorm层
        if self.model_config['norm'] == "pn" and 'freeze' not in self.model_config['dense']:
            x = PartitionedNorm(self.n_domain)([x, domain_indicator])
        if self.model_config['norm'] == "pn" and 'freeze' in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == 'True':
                x = PartitionedNorm(self.n_domain)([x, domain_indicator])
            else:
                x = PartitionedNorm(self.n_domain, trainable=False)([x, domain_indicator])


        # elif self.model_config['norm'] == "dn":
        #     x = DomainNorm(self.n_domain_1)([x, domain_indicator])
        # elif self.model_config['norm'] == "dn2":
        #     x = DomainNorm(self.n_domain_1)([x, domain_indicator])
            # x = DomainNorm(self.n_domain_2)([x, domain_indicator2])
        # elif self.model_config['norm'] == "mdn":
        #     x = MultiDomainNorm(self.n_domain, self.n_domain_1, self.n_domain_2)([x, domain_indicator, domain_indicator1, domain_indicator2])
        # elif self.model_config['norm'] == "mdn_simple":
        #     x = MultiDomainNorm2(self.n_domain_1, self.n_domain_2)([x, domain_indicator1, domain_indicator2])
        elif self.model_config['norm'] == "bn":
            x = layers.BatchNormalization()(x)

        # DomainLoRA
        if self.model_config['dense'] == "dense":
            for h_dim in self.model_config['hidden_dim']:
                x = layers.Dense(h_dim, activation="relu")(x)
        elif self.model_config['dense'] == "mlora":
            lora_r = self.model_config["lora_r"]
            lora_reduce = self.model_config["lora_reduce"]
            dropout_rate = self.model_config["dropout"]
            for i, h_dim in enumerate(self.model_config['hidden_dim']):
                lora_reduce_list = self.model_config["lora_reduce_list"][i]
                lora_weight_list = self.model_config["lora_weight_list"][i]
                x = MLoRAFCN(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce, dropout_rate=dropout_rate,
                             is_finetune=self.train_config['train_from_ckp'],
                             lora_reduce_list=lora_reduce_list,
                             lora_weight_list=lora_weight_list
                             )([x, domain_indicator])
        elif "mlora_skip" in self.model_config['dense']:
            lora_r = self.model_config["lora_r"]
            lora_reduce = self.model_config["lora_reduce"]
            dropout_rate = self.model_config["dropout"]
            h_dim = self.model_config['hidden_dim'][-1]
            x = MLoRAFCN_SKIP(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce, dropout_rate=dropout_rate,
                             is_finetune=self.train_config['train_from_ckp'],
                             n_domain=self.n_domain
                             )([x, domain_indicator])
        elif "mlora_freeze" in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == 'True':
                lora_r = self.model_config["lora_r"]
                lora_reduce = self.model_config["lora_reduce"]
                dropout_rate = self.model_config["dropout"]
                for i, h_dim in enumerate(self.model_config['hidden_dim']):
                    lora_reduce_list = self.model_config["lora_reduce_list"][i] if "lora_reduce_list" in self.model_config else []
                    lora_weight_list = self.model_config["lora_weight_list"][i] if "lora_reduce_list" in self.model_config else []
                    x = MLoRAFCN_PRE(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce, dropout_rate=dropout_rate,
                                 is_finetune=self.train_config['train_from_ckp'],
                                 lora_reduce_list=lora_reduce_list,
                                 lora_weight_list=lora_weight_list
                                 )([x, domain_indicator])
            elif self.model_config['pretrain_judge'] == "False":
                lora_r = self.model_config["lora_r"]
                lora_reduce = self.model_config["lora_reduce"]
                dropout_rate = self.model_config["dropout"]
                for i, h_dim in enumerate(self.model_config['hidden_dim']):
                    lora_reduce_list = self.model_config["lora_reduce_list"][i] if "lora_reduce_list" in self.model_config else []
                    lora_weight_list = self.model_config["lora_weight_list"][i] if "lora_reduce_list" in self.model_config else []
                    x = MLoRAFCN_FREEZE(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce, dropout_rate=dropout_rate,
                                 is_finetune=self.train_config['train_from_ckp'],
                                 lora_reduce_list=lora_reduce_list,
                                 lora_weight_list=lora_weight_list
                                 )([x, domain_indicator])
        elif "mlora_generator" in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == 'True':
                lora_r = self.model_config["lora_r"]
                lora_reduce = self.model_config["lora_reduce"]
                dropout_rate = self.model_config["dropout"]
                for h_dim in self.model_config['hidden_dim']:
                    x = MLoRAFCN_GENERATOR_PRETRAIN(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce,
                                     dropout_rate=dropout_rate,
                                     is_finetune=self.train_config['train_from_ckp'],
                                     )([x, domain_indicator,domain_emb])
            elif self.model_config['pretrain_judge'] == "False":
                lora_r = self.model_config["lora_r"]
                lora_reduce = self.model_config["lora_reduce"]
                dropout_rate = self.model_config["dropout"]
                for h_dim in self.model_config['hidden_dim']:
                    x = MLoRAFCN_GENERATOR(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce,
                                     dropout_rate=dropout_rate,
                                     is_finetune=self.train_config['train_from_ckp'],
                                     )([x, domain_indicator,domain_emb])

        elif "mlora_star_freeze" in self.model_config['dense']:
            assert self.model_config['pretrain_judge'] == 'False',"该方案应提前导入star获得的MLP参数后进行mlora的finetune"
            lora_r = self.model_config["lora_r"]
            lora_reduce = self.model_config["lora_reduce"]
            dropout_rate = self.model_config["dropout"]
            for h_dim in self.model_config['hidden_dim']:
                x = MLoRAFCN_FREEZE(self.n_domain, h_dim, activation="relu", lora_r=lora_r, lora_reduce=lora_reduce,
                                    dropout_rate=dropout_rate,
                                    is_finetune=self.train_config['train_from_ckp'],
                                    )([x, domain_indicator])

        # freeze dense层
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
        # 这里有个坑，因为generator的方案需要传入domain_embedding，之前的方案直接和item、user的embedding合在一起了，导致经过第一个FCN后就没有办法提取domain_embedding了。
        # 因此需要把domain_emb存储下来，每个FCN单独传进去。这里传入的domain_emb本来形状是(?,1,domain_dim)，需要变成(?,8)
        # 不要使用tf.squeeze函数进行删除，不清楚为什么会导致报错： 需要使用keras.layers的操作进行实现

        # Output tensors to a Model must be the output of a TensorFlow `Layer`
        # (thus holding past layer metadata).
        # Found: Tensor("dense/Sigmoid:0", shape=(?, 1), dtype=float32)

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
