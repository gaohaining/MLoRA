import argparse
import json

import tensorflow as tf
from tensorflow.python.keras import backend as K

from model_zoo.DeepCTR import DeepCTR
from model_zoo.DeepCTR import DeepCTR_LORA
from model_zoo.Star import Star

from model_zoo.MLoRA import MLoRA
from utils import MultiDomainDataset
from tensorflow.python.keras import layers, backend, callbacks


def in_name_list(x, name_list):
    for n in name_list:
        if n in x:
            return True
    return False

def layer_matching(prelayer_name,layer_name):
    if prelayer_name == layer_name:
        return True
    elif prelayer_name == "dense" and layer_name == "dense_2":
        return True
    elif prelayer_name == "dense" and layer_name == "dense_1":
        return True
    elif prelayer_name == "partitioned_norm" and layer_name == "partitioned_norm_1":
        return True
    elif prelayer_name == "m_lo_rafcn" and layer_name == "m_lo_rafcn_3":
        return True
    elif prelayer_name == "m_lo_rafcn_1" and layer_name == "m_lo_rafcn_4":
        return True
    elif prelayer_name == "m_lo_rafcn_2" and layer_name == "m_lo_rafcn_5":
        return True
    elif prelayer_name == "star_fcn" and layer_name == "star_fcn_3":
        return True
    elif prelayer_name == "star_fcn_1" and layer_name == "star_fcn_4":
        return True
    elif prelayer_name == "star_fcn_2" and layer_name == "star_fcn_5":
        return True
    else:
        return False

def main(config):
    tf.set_random_seed(config['dataset']['seed'])
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    sess = tf.Session(config=c)
    K.set_session(sess)

    # Load Dataset
    dataset = MultiDomainDataset(config['dataset'])

    # Choose Model
    model = None
    deep_ctr_lora_list = ['mlp_lora', 'wdl_lora', 'nfm_lora', 'autoint_lora','deepfm_lora','dcn_lora','xdeepfm_lora','fibinet_lora','pnn_lora']
    deep_ctr_list = ['mlp_single', 'wdl_single', 'nfm_single', 'autoint_single','deepfm_single','dcn_single','xdeepfm_single','fibinet_single','pnn_single']
    if 'mlora' in config['model']['name']:
        model = MLoRA(dataset, config)
    elif 'star' in config['model']['name']:
        model = Star(dataset, config)
    # deep learning methods
    elif in_name_list(config['model']['name'], deep_ctr_list):
        model = DeepCTR(dataset, config)
    # deep learning methods with MLoRA
    elif in_name_list(config['model']['name'], deep_ctr_lora_list):
        model = DeepCTR_LORA(dataset, config)
    else:
        print("model: {} not found".format(config['model']['name']))

    # get model flops
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    options = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph,options=options)
    print("FLOPS:", flops.total_float_ops)
    sess.close()


    # Train Model

    model.train()

    print("Test Result: ")

    avg_loss, avg_auc, domain_loss, domain_auc = model.val_and_test("test")

    # retrain Model
    if "freeze" in config['model']['dense']:
        graph = tf.get_default_graph()
        if graph.finalized:
            graph._unsafe_unfinalize()

        model.load_model(model.checkpoint_path)
        config['model']['pretrain_judge'] = "False"
        pretrained_model = model
        if 'mlora_freeze' in config['model']['dense']:
            model = MLoRA(dataset, config)
        for pre_layer in pretrained_model.model.layers:
            print(pre_layer.name)
        for layer in model.model.layers:
            print(layer.name)

        # According to the import weights of different finetune policies, the parameters and layer.name that need to be imported are different in different schemes
        if 'mlora_freeze' in config['model']['dense']:
            for pre_layer in pretrained_model.model.layers:
                for layer in model.model.layers:
                    if layer_matching(pre_layer.name, layer.name):
                        if pre_layer.name in ['m_lo_rafcn', 'm_lo_rafcn_1', 'm_lo_rafcn_2']:
                            weights = pre_layer.get_weights()[2:] + pre_layer.get_weights()[:2]
                            model.model.get_layer(layer.name).set_weights(weights)
                        else:
                            model.model.get_layer(layer.name).set_weights(pre_layer.get_weights())

        model.train()


    model.save_result(avg_loss, avg_auc, domain_loss, domain_auc)

    return avg_loss, avg_auc, domain_loss, domain_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Train config file", required=True)
    args = parser.parse_args()
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    main(config)
