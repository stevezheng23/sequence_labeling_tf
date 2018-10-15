import collections

import numpy as np
import tensorflow as tf

from model.seq_crf import *
from util.data_util import *

__all__ = ["TrainModel", "InferModel",
           "create_train_model", "create_infer_model",
           "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "data_pipeline", "word_embedding"))):
    pass

class InferModel(collections.namedtuple("InferModel",
    ("graph", "model", "data_pipeline", "word_embedding", "input_data", "input_text", "input_label"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_data, input_text_data, input_label_data,
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
        
        logger.log_print("# create train text dataset")
        text_dataset = tf.data.Dataset.from_tensor_slices(input_text_data)
        input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
            word_vocab_index, hyperparams.data_text_word_size, hyperparams.data_word_pad, hyperparams.model_word_feat_enable,
            char_vocab_index, hyperparams.data_text_char_size, hyperparams.data_char_pad, hyperparams.model_char_feat_enable)
        
        logger.log_print("# create train label dataset")
        label_dataset = tf.data.Dataset.from_tensor_slices(input_label_data)
        input_label_dataset = create_label_dataset(label_dataset,
            label_vocab_index, hyperparams.data_label_vocab_size, hyperparams.data_label_pad)
        
        logger.log_print("# create train data pipeline")
        data_pipeline = create_data_pipeline(input_text_word_dataset, input_text_char_dataset,
            input_label_dataset, word_vocab_index, hyperparams.data_word_pad, hyperparams.model_word_feat_enable,
            char_vocab_index, hyperparams.data_char_pad, hyperparams.model_char_feat_enable,
            label_vocab_index, hyperparams.data_label_pad, len(input_data), hyperparams.train_batch_size,
            hyperparams.train_random_seed, hyperparams.train_enable_shuffle)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline, word_embedding=word_embed_data)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare infer data")
        (input_data, input_text_data, input_label_data,
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
        
        logger.log_print("# create infer text dataset")
        text_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        text_dataset = tf.data.Dataset.from_tensor_slices(text_placeholder)
        input_text_word_dataset, input_text_char_dataset = create_text_dataset(text_dataset,
            word_vocab_index, hyperparams.data_text_word_size, hyperparams.data_word_pad, hyperparams.model_word_feat_enable,
            char_vocab_index, hyperparams.data_text_char_size, hyperparams.data_char_pad, hyperparams.model_char_feat_enable)
        
        logger.log_print("# create infer label dataset")
        label_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        label_dataset = tf.data.Dataset.from_tensor_slices(label_placeholder)
        input_label_dataset = create_label_dataset(label_dataset,
            label_vocab_index, hyperparams.data_label_vocab_size, hyperparams.data_label_pad)
        
        logger.log_print("# create infer data pipeline")
        data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        data_pipeline = create_dynamic_pipeline(input_text_word_dataset, input_text_char_dataset,
            input_label_dataset, word_vocab_index, hyperparams.data_word_pad, hyperparams.model_word_feat_enable,
            char_vocab_index, hyperparams.data_char_pad, hyperparams.model_char_feat_enable, label_vocab_index,
            hyperparams.data_label_pad, text_placeholder, label_placeholder, data_size_placeholder, batch_size_placeholder)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="infer", scope=hyperparams.model_scope)
        
        return InferModel(graph=graph, model=model, data_pipeline=data_pipeline,
            word_embedding=word_embed_data, input_data=input_data, input_text=input_text_data, input_label=input_label_data)

def get_model_creator(model_type):
    if model_type == "seq_crf":
        model_creator = SequenceCRF
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
