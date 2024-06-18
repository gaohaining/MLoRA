# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364. (https://arxiv.org/abs/1708.05027)
"""
import tensorflow as tf

from deepctr.feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import BiInteractionPooling
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input
from model_zoo.DeepCTR.mlora import Mlora


def NFM(linear_feature_columns, dnn_feature_columns, n_domain, lora_reduce, dnn_hidden_units=(256, 128, 64),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, seed=1024, bi_dropout=0,
        dnn_dropout=0, dnn_activation='relu', task='binary'):


    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())
    domain_input_layer = inputs_list[-1]

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    fm_input = concat_func(sparse_embedding_list, axis=1)
    bi_out = BiInteractionPooling()(fm_input)
    if bi_dropout:
        bi_out = tf.keras.layers.Dropout(bi_dropout)(bi_out, training=None)
    dnn_input = combined_dnn_input([bi_out], dense_value_list)
    # dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)
    dnn_output = Mlora(n_domain, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed,lora_reduce=lora_reduce)([dnn_input,domain_input_layer])

    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)

    final_logit = add_func([linear_logit, dnn_logit])

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
