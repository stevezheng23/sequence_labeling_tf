import collections

import numpy as np
import tensorflow as tf

from model.seq_crf import *
from model.att_crf import *
from model.seq_softmax import *
from util.data_util import *

__all__ = ["TrainModel", "EvalModel", "OnlineModel",
           "create_train_model", "create_eval_model", "create_online_model",
           "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel",
    ("graph", "model", "data_pipeline", "word_embedding", "input_data", "input_text", "input_label"))):
    pass

class EvalModel(collections.namedtuple("EvalModel",
    ("graph", "model", "data_pipeline", "word_embedding", "input_data", "input_text", "input_label"))):
    pass

class OnlineModel(collections.namedtuple("OnlineModel", ("model", "data_pipeline"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_data, input_text_data, input_label_data, input_ext_data,
            word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,  
            char_vocab_size, char_vocab_index, char_vocab_inverted_index,
            label_vocab_size, label_vocab_index, label_vocab_inverted_index) = prepare_sequence_data(logger,
            hyperparams.data_train_sequence_file, hyperparams.data_train_sequence_file_type, hyperparams.data_word_vocab_file,
            hyperparams.data_word_vocab_size, hyperparams.data_word_vocab_threshold, hyperparams.model_word_embed_dim,
            hyperparams.data_embedding_file, hyperparams.data_full_embedding_file, hyperparams.data_word_unk, hyperparams.data_word_pad,
            hyperparams.model_word_feat_enable, hyperparams.model_word_embed_pretrained, hyperparams.data_char_vocab_file,
            hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold, hyperparams.data_char_unk, hyperparams.data_char_pad,
            hyperparams.model_char_feat_enable, hyperparams.data_label_vocab_file, hyperparams.data_label_vocab_size,
            hyperparams.data_label_unk, hyperparams.data_label_pad)
        
        external_data = {}
        if hyperparams.data_pipeline_mode == "dynamic":
            logger.log_print("# create train text dataset")
            text_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            text_dataset = tf.data.Dataset.from_tensor_slices(text_placeholder)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_text_word_size, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_text_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train label dataset")
            label_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            label_dataset = tf.data.Dataset.from_tensor_slices(label_placeholder)
            input_label_dataset = create_label_dataset(label_dataset, label_vocab_index,
                hyperparams.data_label_size, hyperparams.data_label_pad, hyperparams.data_num_parallel)

            logger.log_print("# create train extended dataset")
            ext_placeholder = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
            ext_dataset = tf.data.Dataset.from_tensor_slices(ext_placeholder)
            default_ext_pad = tf.constant(0.0, shape=[], dtype=tf.float32)
            input_ext_dataset = create_ext_dataset(ext_dataset,
                hyperparams.data_ext_max_size, hyperparams.model_ext_embed_dim, default_ext_pad,
                hyperparams.model_ext_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train data pipeline")
            data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            data_pipeline = create_dynamic_pipeline(input_text_word_dataset,
                input_text_char_dataset, input_label_dataset, input_ext_dataset,
                word_vocab_size, word_vocab_index, hyperparams.data_word_pad, hyperparams.model_word_feat_enable,
                char_vocab_size, char_vocab_index, hyperparams.data_char_pad, hyperparams.model_char_feat_enable,
                label_vocab_index, label_vocab_inverted_index, hyperparams.data_label_pad, hyperparams.model_ext_feat_enable,
                hyperparams.train_random_seed, hyperparams.train_enable_shuffle, hyperparams.train_shuffle_buffer_size, 
                text_placeholder, label_placeholder, ext_placeholder, data_size_placeholder, batch_size_placeholder)
        else:
            if word_embed_data is not None:
                external_data["word_embedding"] = word_embed_data

            logger.log_print("# create train text dataset")
            text_dataset = tf.data.Dataset.from_tensor_slices(input_text_data)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_text_word_size, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_text_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train label dataset")
            label_dataset = tf.data.Dataset.from_tensor_slices(input_label_data)
            input_label_dataset = create_label_dataset(label_dataset, label_vocab_index,
                hyperparams.data_label_size, hyperparams.data_label_pad, hyperparams.data_num_parallel)

            logger.log_print("# create train extended dataset")
            ext_dataset = tf.data.Dataset.from_tensor_slices(input_ext_data)
            default_ext_pad = tf.constant(0.0, shape=[], dtype=tf.float32)
            input_ext_dataset = create_ext_dataset(ext_dataset,
                hyperparams.data_ext_max_size, hyperparams.model_ext_embed_dim, default_ext_pad,
                hyperparams.model_ext_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create train data pipeline")
            data_pipeline = create_data_pipeline(input_text_word_dataset,
                input_text_char_dataset, input_label_dataset, input_ext_dataset,
                word_vocab_size, word_vocab_index, hyperparams.data_word_pad, hyperparams.model_word_feat_enable,
                char_vocab_size, char_vocab_index, hyperparams.data_char_pad, hyperparams.model_char_feat_enable,
                label_vocab_index, label_vocab_inverted_index, hyperparams.data_label_pad, hyperparams.model_ext_feat_enable,
                hyperparams.train_random_seed, hyperparams.train_enable_shuffle, hyperparams.train_shuffle_buffer_size,
                len(input_data), hyperparams.train_batch_size)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            external_data=external_data, mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline,
            word_embedding=word_embed_data, input_data=input_data, input_text=input_text_data, input_label=input_label_data)

def create_eval_model(logger,
                      hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare eval data")
        (input_data, input_text_data, input_label_data, input_ext_data,
            word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,  
            char_vocab_size, char_vocab_index, char_vocab_inverted_index,
            label_vocab_size, label_vocab_index, label_vocab_inverted_index) = prepare_sequence_data(logger,
            hyperparams.data_eval_sequence_file, hyperparams.data_eval_sequence_file_type, hyperparams.data_word_vocab_file,
            hyperparams.data_word_vocab_size, hyperparams.data_word_vocab_threshold, hyperparams.model_word_embed_dim,
            hyperparams.data_embedding_file, hyperparams.data_full_embedding_file, hyperparams.data_word_unk, hyperparams.data_word_pad,
            hyperparams.model_word_feat_enable, hyperparams.model_word_embed_pretrained, hyperparams.data_char_vocab_file,
            hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold, hyperparams.data_char_unk, hyperparams.data_char_pad,
            hyperparams.model_char_feat_enable, hyperparams.data_label_vocab_file, hyperparams.data_label_vocab_size,
            hyperparams.data_label_unk, hyperparams.data_label_pad)
        
        external_data = {}
        if hyperparams.data_pipeline_mode == "dynamic":
            logger.log_print("# create eval text dataset")
            text_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            text_dataset = tf.data.Dataset.from_tensor_slices(text_placeholder)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_text_word_size, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_text_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create eval label dataset")
            label_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
            label_dataset = tf.data.Dataset.from_tensor_slices(label_placeholder)
            input_label_dataset = create_label_dataset(label_dataset, label_vocab_index,
                hyperparams.data_label_size, hyperparams.data_label_pad, hyperparams.data_num_parallel)

            logger.log_print("# create eval extended dataset")
            ext_placeholder = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
            ext_dataset = tf.data.Dataset.from_tensor_slices(ext_placeholder)
            default_ext_pad = tf.constant(0.0, shape=[], dtype=tf.float32)
            input_ext_dataset = create_ext_dataset(ext_dataset,
                hyperparams.data_ext_max_size, hyperparams.model_ext_embed_dim, default_ext_pad,
                hyperparams.model_ext_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create eval data pipeline")
            data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
            data_pipeline = create_dynamic_pipeline(input_text_word_dataset,
                input_text_char_dataset, input_label_dataset, input_ext_dataset,
                word_vocab_size, word_vocab_index, hyperparams.data_word_pad, hyperparams.model_word_feat_enable,
                char_vocab_size, char_vocab_index, hyperparams.data_char_pad, hyperparams.model_char_feat_enable,
                label_vocab_index, label_vocab_inverted_index, hyperparams.data_label_pad, hyperparams.model_ext_feat_enable,
                None, False, 0, text_placeholder, label_placeholder, ext_placeholder, data_size_placeholder, batch_size_placeholder)
        else:
            if word_embed_data is not None:
                external_data["word_embedding"] = word_embed_data

            logger.log_print("# create eval text dataset")
            text_dataset = tf.data.Dataset.from_tensor_slices(input_text_data)
            input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
                word_vocab_index, hyperparams.data_text_word_size, hyperparams.data_word_pad,
                hyperparams.model_word_feat_enable, char_vocab_index, hyperparams.data_text_char_size,
                hyperparams.data_char_pad, hyperparams.model_char_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create eval label dataset")
            label_dataset = tf.data.Dataset.from_tensor_slices(input_label_data)
            input_label_dataset = create_label_dataset(label_dataset, label_vocab_index,
                hyperparams.data_label_size, hyperparams.data_label_pad, hyperparams.data_num_parallel)

            logger.log_print("# create eval extended dataset")
            ext_dataset = tf.data.Dataset.from_tensor_slices(input_ext_data)
            default_ext_pad = tf.constant(0.0, shape=[], dtype=tf.float32)
            input_ext_dataset = create_ext_dataset(ext_dataset,
                hyperparams.data_ext_max_size, hyperparams.model_ext_embed_dim, default_ext_pad,
                hyperparams.model_ext_feat_enable, hyperparams.data_num_parallel)

            logger.log_print("# create eval data pipeline")
            data_pipeline = create_data_pipeline(input_text_word_dataset,
                input_text_char_dataset, input_label_dataset, input_ext_dataset,
                word_vocab_size, word_vocab_index, hyperparams.data_word_pad, hyperparams.model_word_feat_enable,
                char_vocab_size, char_vocab_index, hyperparams.data_char_pad, hyperparams.model_char_feat_enable,
                label_vocab_index, label_vocab_inverted_index, hyperparams.data_label_pad, hyperparams.model_ext_feat_enable,
                None, False, 0, len(input_data), hyperparams.train_eval_batch_size)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            external_data=external_data, mode="eval", scope=hyperparams.model_scope)
        
        return EvalModel(graph=graph, model=model, data_pipeline=data_pipeline,
            word_embedding=word_embed_data, input_data=input_data, input_text=input_text_data, input_label=input_label_data)

def create_online_model(logger,
                        hyperparams):
    logger.log_print("# prepare online data")
    (word_embed_data, word_vocab_size, word_vocab_index,
        word_vocab_inverted_index, char_vocab_size, char_vocab_index,
        char_vocab_inverted_index) = prepare_text_data(logger, None, hyperparams.data_word_vocab_file,
        hyperparams.data_word_vocab_size, hyperparams.data_word_vocab_threshold, hyperparams.model_word_embed_dim,
        hyperparams.data_embedding_file, hyperparams.data_full_embedding_file, hyperparams.data_word_unk,
        hyperparams.data_word_pad, hyperparams.model_word_feat_enable, hyperparams.model_word_embed_pretrained,
        hyperparams.data_char_vocab_file, hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold,
        hyperparams.data_char_unk, hyperparams.data_char_pad, hyperparams.model_char_feat_enable)
    
    label_vocab_size, label_vocab_index, label_vocab_inverted_index = prepare_label_data(logger, None,
        hyperparams.data_label_vocab_file, hyperparams.data_label_vocab_size, hyperparams.data_label_unk, hyperparams.data_label_pad)
    
    external_data={}
    if word_embed_data is not None:
        external_data["word_embedding"] = word_embed_data
    
    logger.log_print("# create online data pipeline")
    data_pipeline = create_online_pipeline(hyperparams.data_external_index_enable,
        word_vocab_size, word_vocab_index, hyperparams.data_text_word_size, hyperparams.data_word_pad,
        hyperparams.model_word_feat_enable, char_vocab_size, char_vocab_index, hyperparams.data_text_char_size,
        hyperparams.data_char_pad, hyperparams.model_char_feat_enable, label_vocab_inverted_index,
        hyperparams.data_ext_max_size, hyperparams.model_ext_embed_dim, hyperparams.model_ext_feat_enable)

    model_creator = get_model_creator(hyperparams.model_type)
    model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
        external_data=external_data, mode="online", scope=hyperparams.model_scope)

    return OnlineModel(model=model, data_pipeline=data_pipeline)

def get_model_creator(model_type):
    if model_type == "seq_softmax":
        model_creator = SequenceSoftmax
    elif model_type == "seq_crf":
        model_creator = SequenceCRF
    elif model_type == "att_crf":
        model_creator = AttentionCRF
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(model_type))
    
    return model_creator

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model,
               ckpt_file,
               ckpt_type):
    with model.graph.as_default():
        model.model.restore(sess, ckpt_file, ckpt_type)
