import argparse
import codecs
import json
import math
import os.path

import numpy as np
import tensorflow as tf

__all__ = ["create_default_hyperparams", "load_hyperparams",
           "generate_search_lookup", "search_hyperparams", "create_hyperparams_file"]

def create_default_hyperparams(config_type):
    """create default hyperparameters"""
    if config_type == "seq_crf":
        hyperparams = tf.contrib.training.HParams(
            data_train_sequence_file="",
            data_train_sequence_file_type="",
            data_eval_sequence_file="",
            data_eval_sequence_file_type="",
            data_embedding_file="",
            data_full_embedding_file="",
            data_text_word_size=50,
            data_text_char_size=16,
            data_label_size=50,
            data_word_vocab_file="",
            data_word_vocab_size=30000,
            data_word_vocab_threshold=0,
            data_word_unk="<unk>",
            data_word_pad="<pad>",
            data_char_vocab_file="",
            data_char_vocab_size=100,
            data_char_vocab_threshold=0,
            data_char_unk="*",
            data_char_pad="#",
            data_label_vocab_file="",
            data_label_vocab_size=10,
            data_label_unk="O",
            data_label_pad="P",
            data_external_index_enable=False,
            data_log_output_dir="",
            data_result_output_dir="",
            train_random_seed=100,
            train_enable_shuffle=True,
            train_shuffle_buffer_size=15000,
            train_batch_size=20,
            train_eval_batch_size=100,
            train_eval_metric=["precision", "recall", "f1"],
            train_num_epoch=3,
            train_model_version="",
            train_model_output_dir="",
            train_ckpt_output_dir="",
            train_summary_output_dir="",
            train_step_per_stat=10,
            train_step_per_ckpt=1000,
            train_step_per_eval=1000,
            train_clip_norm=5.0,
            train_ema_enable=True,
            train_ema_decay_rate=0.999,
            train_regularization_enable=False,
            train_regularization_type="l2",
            train_regularization_scale=5e-8,
            train_optimizer_type="adam",
            train_optimizer_learning_rate=0.001,
            train_optimizer_warmup_enable=False,
            train_optimizer_warmup_mode="exponential_warmup",
            train_optimizer_warmup_rate=0.01,
            train_optimizer_warmup_end_step=1000,
            train_optimizer_decay_enable=False,
            train_optimizer_decay_mode="exponential_decay",
            train_optimizer_decay_rate=0.95,
            train_optimizer_decay_step=500,
            train_optimizer_decay_start_step=1500,
            train_optimizer_momentum_beta=0.9,
            train_optimizer_rmsprop_beta=0.999,
            train_optimizer_rmsprop_epsilon=1e-8,
            train_optimizer_adadelta_rho=0.95,
            train_optimizer_adadelta_epsilon=1e-8,
            train_optimizer_adagrad_init_accumulator=0.1,
            train_optimizer_adam_beta_1=0.9,
            train_optimizer_adam_beta_2=0.999,
            train_optimizer_adam_epsilon=1e-08,
            model_type="seq_crf",
            model_scope="sequence_labeling",
            model_word_embed_dim=300,
            model_word_dropout=0.5,
            model_word_embed_pretrained=True,
            model_word_feat_trainable=False,
            model_word_feat_enable=True,
            model_char_embed_dim=16,
            model_char_unit_dim=100,
            model_char_window_size=[3],
            model_char_hidden_activation="relu",
            model_char_dropout=0.5,
            model_char_pooling_type="max",
            model_char_feat_trainable=True,
            model_char_feat_enable=True,
            model_fusion_type="concate",
            model_fusion_num_layer=1,
            model_fusion_unit_dim=400,
            model_fusion_hidden_activation="relu",
            model_fusion_dropout=0.0,
            model_fusion_trainable=False,
            model_sequence_num_layer=1,
            model_sequence_unit_dim=100,
            model_sequence_cell_type="lstm",
            model_sequence_hidden_activation="tanh",
            model_sequence_dropout=0.0,
            model_sequence_forget_bias=1.0,
            model_sequence_residual_connect=False,
            model_sequence_trainable=True,
            model_labeling_unit_dim=9,
            model_labeling_dropout=0.5,
            model_labeling_trainable=True,
            device_num_gpus=1,
            device_default_gpu_id=0,
            device_log_device_placement=False,
            device_allow_soft_placement=False,
            device_allow_growth=False,
            device_per_process_gpu_memory_fraction=0.8
        )
    else:
        raise ValueError("unsupported config type {0}".format(config_type))
    
    return hyperparams

def load_hyperparams(config_file):
    """load hyperparameters from config file"""
    if tf.gfile.Exists(config_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(config_file, "rb")) as file:
            hyperparams_dict = json.load(file)
            hyperparams = create_default_hyperparams(hyperparams_dict["model_type"])
            hyperparams.override_from_dict(hyperparams_dict)
            
            return hyperparams
    else:
        raise FileNotFoundError("config file not found")

def generate_search_lookup(search,
                           search_lookup=None):
    search_lookup = search_lookup if search_lookup else {}
    search_type = search["stype"]
    data_type = search["dtype"]
    
    if search_type == "uniform":
        range_start = search["range"][0]
        range_end = search["range"][1]
        if data_type == "int":
            search_sample = np.random.randint(range_start, range_end)
        elif data_type == "float":
            search_sample = (range_end - range_start) * np.random.random_sample() + range_start
        else:
            raise ValueError("unsupported data type {0}".format(data_type))
    elif search_type == "log":
        range_start = math.log(search["range"][0], 10)
        range_end = math.log(search["range"][1], 10)
        if data_type == "float":
            search_sample = math.pow(10, (range_end - range_start) * np.random.random_sample() + range_start)
        else:
            raise ValueError("unsupported data type {0}".format(data_type))
    elif search_type == "discrete":
        search_set = search["set"]
        search_index = np.random.choice(len(search_set))
        search_sample = search_set[search_index]
    elif search_type == "lookup":
        search_key = search["key"]
        if search_key in search_lookup:
            search_sample = search_lookup[search_key]
        else:
            raise ValueError("search key {0} doesn't exist in look-up table".format(search_key))
    else:
        raise ValueError("unsupported search type {0}".format(search_type))
    
    data_scale = search["scale"] if "scale" in search else 1.0
    data_shift = search["shift"] if "shift" in search else 0.0
    
    if data_type == "int":
        search_sample = int(data_scale * search_sample + data_shift)
    elif data_type == "float":
        search_sample = float(data_scale * search_sample + data_shift)
    elif data_type == "string":
        search_sample = str(search_sample)
    elif data_type == "boolean":
        search_sample = bool(search_sample)
    elif data_type == "list":
        search_sample = list(search_sample)
    else:
        raise ValueError("unsupported data type {0}".format(data_type))
    
    return search_sample

def search_hyperparams(hyperparams,
                       config_file,
                       num_group,
                       random_seed):
    """search hyperparameters based on search config"""
    if tf.gfile.Exists(config_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(config_file, "rb")) as file:
            hyperparams_group = []
            np.random.seed(random_seed)
            search_setting = json.load(file)
            hyperparams_search_setting = search_setting["hyperparams"]
            variables_search_setting = search_setting["variables"]
            for i in range(num_group):
                variables_search_lookup = {}
                for key in variables_search_setting.keys():
                    variables_search = variables_search_setting[key]
                    variables_search_lookup[key] = generate_search_lookup(variables_search)
                hyperparams_search_lookup = {}
                for key in hyperparams_search_setting.keys():
                    hyperparams_search = hyperparams_search_setting[key]
                    hyperparams_search_lookup[key] = generate_search_lookup(hyperparams_search, variables_search_lookup)
                
                hyperparams_sample = tf.contrib.training.HParams(hyperparams.to_proto())
                hyperparams_sample.override_from_dict(hyperparams_search_lookup)
                hyperparams_group.append(hyperparams_sample)
            
            return hyperparams_group
    else:
        raise FileNotFoundError("config file not found")

def create_hyperparams_file(hyperparams_group, config_dir):
    """create config files from groups of hyperparameters"""
    if not tf.gfile.Exists(config_dir):
        tf.gfile.MakeDirs(config_dir)
    
    for i in range(len(hyperparams_group)):
        config_file = os.path.join(config_dir, "config_hyperparams_{0}.json".format(i))
        with codecs.getwriter("utf-8")(tf.gfile.GFile(config_file, "w")) as file:
            hyperparam_dict = hyperparams_group[i].values()
            hyperparams_json = json.dumps(hyperparam_dict, indent=4)
            file.write(hyperparams_json)
