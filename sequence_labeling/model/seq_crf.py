import collections
import functools
import os.path
import operator
import time

import numpy as np
import tensorflow as tf

from functools import reduce

from util.default_util import *
from util.sequence_labeling_util import *
from util.layer_util import *

from model.base_model import *

__all__ = ["SequenceCRF"]

class SequenceCRF(BaseModel):
    """sequence crf model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 external_data,
                 mode="train",
                 scope="seq_crf"):
        """initialize sequence crf model"""
        super(SequenceCRF, self).__init__(logger=logger, hyperparams=hyperparams,
            data_pipeline=data_pipeline, external_data=external_data, mode=mode, scope=scope)
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            """get batch input from data pipeline"""
            text_word = self.data_pipeline.input_text_word
            text_word_mask = self.data_pipeline.input_text_word_mask
            text_char = self.data_pipeline.input_text_char
            text_char_mask = self.data_pipeline.input_text_char_mask
            label_inverted_index = self.data_pipeline.label_inverted_index
            self.word_vocab_size = self.data_pipeline.word_vocab_size
            self.char_vocab_size = self.data_pipeline.char_vocab_size
            self.sequence_length = tf.cast(tf.reduce_sum(text_word_mask, axis=[-1, -2]), dtype=tf.int32)
            
            """build graph for sequence crf model"""
            self.logger.log_print("# build graph")
            predict, predict_mask, transition_matrix = self._build_graph(text_word, text_word_mask, text_char, text_char_mask)
            masked_predict = predict * predict_mask
            self.index_predict, _ = tf.contrib.crf.crf_decode(masked_predict, transition_matrix, self.sequence_length)
            self.text_predict = label_inverted_index.lookup(tf.cast(self.index_predict, dtype=tf.int64))
            
            self.variable_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.variable_lookup = {v.op.name: v for v in self.variable_list}
            
            self.transferable_list = tf.get_collection(TRANSFERABLE_VARIABLES)
            self.transferable_lookup = {v.op.name: v for v in self.transferable_list}
            
            if self.hyperparams.train_ema_enable == True:
                self.ema = self._get_exponential_moving_average(self.global_step)
                self.variable_lookup = {self.ema.average_name(v): v for v in self.variable_list}
                self.transferable_lookup = {self.ema.average_name(v): v for v in self.transferable_list}
            
            if self.mode == "train":
                self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                    initializer=tf.zeros_initializer, trainable=False)
                
                label = tf.squeeze(self.data_pipeline.input_label, axis=-1)
                label_mask = tf.squeeze(self.data_pipeline.input_label_mask, axis=-1)
                masked_label = tf.cast(label * label_mask, dtype=tf.int32)
                
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                self.train_loss = self._compute_loss(masked_label, masked_predict, self.sequence_length, transition_matrix)
                
                if self.hyperparams.train_regularization_enable == True:
                    regularization_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    regularization_loss = tf.contrib.layers.apply_regularization(self.regularizer, regularization_variables)
                    self.train_loss = self.train_loss + regularization_loss
                
                """apply learning rate warm-up & decay"""
                self.logger.log_print("# setup initial learning rate mechanism")
                self.initial_learning_rate = tf.constant(self.hyperparams.train_optimizer_learning_rate)
                
                if self.hyperparams.train_optimizer_warmup_enable == True:
                    self.logger.log_print("# setup learning rate warm-up mechanism")
                    self.warmup_learning_rate = self._apply_learning_rate_warmup(self.initial_learning_rate)
                else:
                    self.warmup_learning_rate = self.initial_learning_rate
                
                if self.hyperparams.train_optimizer_decay_enable == True:
                    self.logger.log_print("# setup learning rate decay mechanism")
                    self.decayed_learning_rate = self._apply_learning_rate_decay(self.warmup_learning_rate)
                else:
                    self.decayed_learning_rate = self.warmup_learning_rate
                
                self.learning_rate = self.decayed_learning_rate
                
                """initialize optimizer"""
                self.logger.log_print("# setup training optimizer")
                self.optimizer = self._initialize_optimizer(self.learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.opt_op, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                if self.hyperparams.train_ema_enable == True:
                    with tf.control_dependencies([self.opt_op]):
                        self.update_op = self.ema.apply(self.variable_list)
                        self.variable_lookup = {self.ema.average_name(v): self.ema.average(v) for v in self.variable_list}
                else:
                    self.update_op = self.opt_op
                
                """create train summary"""
                self.train_summary = self._get_train_summary()
            
            if self.mode == "online":
                """create model builder"""
                if not tf.gfile.Exists(self.hyperparams.train_model_output_dir):
                    tf.gfile.MakeDirs(self.hyperparams.train_model_output_dir)
                
                model_version = "{0}.{1}".format(self.hyperparams.train_model_version, time.time())
                self.model_dir = os.path.join(self.hyperparams.train_model_output_dir, model_version)
                self.model_builder = tf.saved_model.builder.SavedModelBuilder(self.model_dir)
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_debug_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "debug")
            self.ckpt_epoch_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "epoch")
            self.ckpt_transfer_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "transfer")
            
            if not tf.gfile.Exists(self.ckpt_debug_dir):
                tf.gfile.MakeDirs(self.ckpt_debug_dir)
            
            if not tf.gfile.Exists(self.ckpt_epoch_dir):
                tf.gfile.MakeDirs(self.ckpt_epoch_dir)
            
            if not tf.gfile.Exists(self.ckpt_transfer_dir):
                tf.gfile.MakeDirs(self.ckpt_transfer_dir)
            
            self.ckpt_debug_name = os.path.join(self.ckpt_debug_dir, "model_debug_ckpt")
            self.ckpt_epoch_name = os.path.join(self.ckpt_epoch_dir, "model_epoch_ckpt")
            
            self.ckpt_debug_saver = tf.train.Saver(self.variable_lookup)
            self.ckpt_epoch_saver = tf.train.Saver(self.variable_lookup, max_to_keep=self.hyperparams.train_num_epoch)
            self.ckpt_transfer_saver = (tf.train.Saver(self.transferable_lookup)
                if any(self.transferable_lookup) else tf.train.Saver(self.variable_lookup))
    
    def _build_representation_layer(self,
                                    text_word,
                                    text_word_mask,
                                    text_char,
                                    text_char_mask):
        """build representation layer for sequence crf model"""
        word_embed_dim = self.hyperparams.model_word_embed_dim
        word_dropout = self.hyperparams.model_word_dropout if self.mode == "train" else 0.0
        word_embed_pretrained = self.hyperparams.model_word_embed_pretrained
        word_feat_feedable = False if self.mode == "online" else True
        word_feat_trainable = self.hyperparams.model_word_feat_trainable
        word_feat_enable = self.hyperparams.model_word_feat_enable
        char_embed_dim = self.hyperparams.model_char_embed_dim
        char_unit_dim = self.hyperparams.model_char_unit_dim
        char_window_size = self.hyperparams.model_char_window_size
        char_hidden_activation = self.hyperparams.model_char_hidden_activation
        char_dropout = self.hyperparams.model_char_dropout if self.mode == "train" else 0.0
        char_pooling_type = self.hyperparams.model_char_pooling_type
        char_feat_trainable = self.hyperparams.model_char_feat_trainable
        char_feat_enable = self.hyperparams.model_char_feat_enable
        fusion_type = self.hyperparams.model_fusion_type
        fusion_num_layer = self.hyperparams.model_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_fusion_hidden_activation
        fusion_dropout = self.hyperparams.model_fusion_dropout if self.mode == "train" else 0.0
        fusion_trainable = self.hyperparams.model_fusion_trainable
        
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            text_feat_list = []
            text_feat_mask_list = []
            
            if word_feat_enable == True:
                self.logger.log_print("# build word-level representation layer")
                word_feat_layer = WordFeat(vocab_size=self.word_vocab_size, embed_dim=word_embed_dim,
                    dropout=word_dropout, pretrained=word_embed_pretrained, embedding=self.word_embedding,
                    num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, feedable=word_feat_feedable, trainable=word_feat_trainable)
                
                (text_word_feat,
                    text_word_feat_mask) = word_feat_layer(text_word, text_word_mask)
                text_feat_list.append(text_word_feat)
                text_feat_mask_list.append(text_word_feat_mask)
                
                word_unit_dim = word_embed_dim
                self.word_embedding_placeholder = word_feat_layer.get_embedding_placeholder()
            else:
                word_unit_dim = 0
                self.word_embedding_placeholder = None
            
            if char_feat_enable == True:
                self.logger.log_print("# build char-level representation layer")
                char_feat_layer = CharFeat(vocab_size=self.char_vocab_size, embed_dim=char_embed_dim, unit_dim=char_unit_dim,
                    window_size=char_window_size, activation=char_hidden_activation, pooling_type=char_pooling_type,
                    dropout=char_dropout, num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id,
                    regularizer=self.regularizer, random_seed=self.random_seed, trainable=char_feat_trainable)
                
                (text_char_feat,
                    text_char_feat_mask) = char_feat_layer(text_char, text_char_mask)
                
                text_feat_list.append(text_char_feat)
                text_feat_mask_list.append(text_char_feat_mask)
            else:
                char_unit_dim = 0
            
            feat_unit_dim = word_unit_dim + char_unit_dim
            feat_fusion_layer = FusionModule(input_unit_dim=feat_unit_dim, output_unit_dim=fusion_unit_dim,
                fusion_type=fusion_type, num_layer=fusion_num_layer, activation=fusion_hidden_activation,
                dropout=fusion_dropout, num_gpus=self.num_gpus, default_gpu_id=self.default_gpu_id,
                regularizer=self.regularizer, random_seed=self.random_seed, trainable=fusion_trainable)
            
            text_feat, text_feat_mask = feat_fusion_layer(text_feat_list, text_feat_mask_list)
        
        return text_feat, text_feat_mask
    
    def _build_modeling_layer(self,
                              text_feat,
                              text_feat_mask):
        """build modeling layer for sequence crf model"""
        sequence_num_layer = self.hyperparams.model_sequence_num_layer
        sequence_unit_dim = self.hyperparams.model_sequence_unit_dim
        sequence_cell_type = self.hyperparams.model_sequence_cell_type
        sequence_hidden_activation = self.hyperparams.model_sequence_hidden_activation
        sequence_dropout = self.hyperparams.model_sequence_dropout if self.mode == "train" else 0.0
        sequence_forget_bias = self.hyperparams.model_sequence_forget_bias
        sequence_residual_connect = self.hyperparams.model_sequence_residual_connect
        sequence_trainable = self.hyperparams.model_sequence_trainable
        labeling_unit_dim = self.hyperparams.model_labeling_unit_dim
        labeling_dropout = self.hyperparams.model_labeling_dropout
        labeling_trainable = self.hyperparams.model_labeling_trainable
        labeling_transferable = self.hyperparams.model_labeling_transferable
        
        with tf.variable_scope("modeling", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build sequence modeling layer")
            sequence_modeling_layer = create_recurrent_layer("bi", sequence_num_layer, sequence_unit_dim,
                sequence_cell_type, sequence_hidden_activation, sequence_dropout, sequence_forget_bias,
                sequence_residual_connect, None, self.num_gpus, self.default_gpu_id, self.random_seed, sequence_trainable)
            
            (text_sequence_modeling, text_sequence_modeling_mask,
                _, _) = sequence_modeling_layer(text_feat, text_feat_mask)
            
            if labeling_transferable == False:
                pre_labeling_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            
            labeling_modeling_layer = create_dense_layer("single", 1, labeling_unit_dim, 1, "", [labeling_dropout], None,
                False, False, True, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, labeling_trainable)
            
            (text_labeling_modeling,
                text_labeling_modeling_mask) = labeling_modeling_layer(text_sequence_modeling, text_sequence_modeling_mask)
            
            text_modeling = text_labeling_modeling
            text_modeling_mask = text_labeling_modeling_mask
            
            weight_initializer = create_variable_initializer("glorot_uniform", self.random_seed)
            text_modeling_matrix = tf.get_variable("transition_matrix", shape=[labeling_unit_dim, labeling_unit_dim],
                initializer=weight_initializer, regularizer=self.regularizer, trainable=labeling_trainable, dtype=tf.float32)
            
            if labeling_transferable == False:
                post_labeling_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                [tf.add_to_collection(TRANSFERABLE_VARIABLES, v) for v in post_labeling_variables if v in pre_labeling_variables]
        
        return text_modeling, text_modeling_mask, text_modeling_matrix
     
    def _build_graph(self,
                     text_word,
                     text_word_mask,
                     text_char,
                     text_char_mask):
        """build graph for sequence crf model"""
        with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
            """build representation layer for sequence crf model"""
            text_feat, text_feat_mask = self._build_representation_layer(text_word,
                text_word_mask, text_char, text_char_mask)
            
            """build modeling layer for sequence crf model"""
            (text_modeling, text_modeling_mask,
                text_modeling_matrix) = self._build_modeling_layer(text_feat, text_feat_mask)
            
            predict = text_modeling
            predict_mask = text_modeling_mask
            transition_matrix = text_modeling_matrix
        
        return predict, predict_mask, transition_matrix
    
    def _compute_loss(self,
                      label,
                      predict,
                      sequence_length,
                      transition_matrix):
        """compute optimization loss"""
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(predict, label, sequence_length, transition_matrix)
        loss = tf.reduce_mean(-1.0 * log_likelihood)
        
        return loss
    
    def build(self,
              sess):
        """build saved model for sequence crf model"""
        external_index_enable = self.hyperparams.data_external_index_enable
        if external_index_enable == True:
            input_word = tf.saved_model.utils.build_tensor_info(self.data_pipeline.input_word_placeholder)
            input_char = tf.saved_model.utils.build_tensor_info(self.data_pipeline.input_char_placeholder)
            output_predict = tf.saved_model.utils.build_tensor_info(self.index_predict)
            output_sequence_length = tf.saved_model.utils.build_tensor_info(self.sequence_length)

            predict_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'input_word': input_word,
                    'input_char': input_char,
                },
                outputs={
                    'output_predict': output_predict,
                    'output_sequence_length': output_sequence_length
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        else:
            input_text = tf.saved_model.utils.build_tensor_info(self.data_pipeline.input_text_placeholder)
            output_predict = tf.saved_model.utils.build_tensor_info(self.text_predict)
            output_sequence_length = tf.saved_model.utils.build_tensor_info(self.sequence_length)

            predict_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs={ 'input_text': input_text },
                outputs={
                    'output_predict': output_predict,
                    'output_sequence_length': output_sequence_length
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        
        self.model_builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                predict_signature
            },
            clear_devices=True,
            main_op=tf.tables_initializer())
        
        self.model_builder.save(as_text=False)
    
    def save(self,
             sess,
             global_step,
             save_mode):
        """save checkpoint for sequence crf model"""
        if save_mode == "debug":
            self.ckpt_debug_saver.save(sess, self.ckpt_debug_name, global_step=global_step)
        elif save_mode == "epoch":
            self.ckpt_epoch_saver.save(sess, self.ckpt_epoch_name, global_step=global_step)
        else:
            raise ValueError("unsupported save mode {0}".format(save_mode))
    
    def restore(self,
                sess,
                ckpt_file,
                ckpt_type):
        """restore sequence crf model from checkpoint"""
        if ckpt_file is None:
            raise FileNotFoundError("checkpoint file doesn't exist")
        
        if ckpt_type == "debug":
            self.ckpt_debug_saver.restore(sess, ckpt_file)
        elif ckpt_type == "epoch":
            self.ckpt_epoch_saver.restore(sess, ckpt_file)
        elif ckpt_type == "transfer":
            self.ckpt_transfer_saver.restore(sess, ckpt_file)
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_latest_ckpt(self,
                        ckpt_type):
        """get the latest checkpoint for sequence crf model"""
        if ckpt_type == "debug":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_debug_dir)
        elif ckpt_type == "epoch":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_epoch_dir)
        elif ckpt_type == "transfer":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_transfer_dir)
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
        
        if ckpt_file is None:
            raise FileNotFoundError("latest checkpoint file doesn't exist")
        
        return ckpt_file
    
    def get_ckpt_list(self,
                      ckpt_type):
        """get checkpoint list for sequence crf model"""
        if ckpt_type == "debug":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_debug_dir)
        elif ckpt_type == "epoch":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_epoch_dir)
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
        
        if ckpt_state is None:
            raise FileNotFoundError("checkpoint files doesn't exist")
        
        return ckpt_state.all_model_checkpoint_paths

class WordFeat(object):
    """word-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 dropout,
                 pretrained,
                 embedding=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 feedable=True,
                 trainable=True,
                 scope="word_feat"):
        """initialize word-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.pretrained = pretrained
        self.embedding = embedding
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.feedable = feedable
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size, self.embed_dim, self.pretrained, self.embedding,
                self.num_gpus, self.default_gpu_id, None, self.random_seed, self.feedable, self.trainable)
            
            self.dropout_layer = create_dropout_layer(self.dropout, self.num_gpus, self.default_gpu_id, self.random_seed)
    
    def __call__(self,
                 input_word,
                 input_word_mask):
        """call word-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_word_embedding_mask = input_word_mask
            input_word_embedding = tf.squeeze(self.embedding_layer(input_word), axis=-2)
            
            (input_word_dropout,
                input_word_dropout_mask) = self.dropout_layer(input_word_embedding, input_word_embedding_mask)
            
            input_word_feat = input_word_dropout
            input_word_feat_mask = input_word_dropout_mask
        
        return input_word_feat, input_word_feat_mask
    
    def get_embedding_placeholder(self):
        """get word-level embedding placeholder"""
        return self.embedding_layer.get_embedding_placeholder()

class CharFeat(object):
    """char-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 unit_dim,
                 window_size,
                 activation,
                 pooling_type,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="char_feat"):
        """initialize char-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.activation = activation
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size, self.embed_dim, False, None,
                self.num_gpus, self.default_gpu_id, None, self.random_seed, False, self.trainable)
            
            self.conv_layer = create_convolution_layer("stacked_multi_1d", 1, self.embed_dim,
                self.unit_dim, self.window_size, 1, "SAME", self.activation, [self.dropout], None,
                False, False, True, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            self.dropout_layer = create_dropout_layer(self.dropout, self.num_gpus, self.default_gpu_id, self.random_seed)
            
            self.pooling_layer = create_pooling_layer(self.pooling_type, 1, 1, self.num_gpus, self.default_gpu_id)
    
    def __call__(self,
                 input_char,
                 input_char_mask):
        """call char-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_char_embedding_mask = tf.expand_dims(input_char_mask, axis=-1)
            input_char_embedding = self.embedding_layer(input_char)
            
            (input_char_dropout,
                input_char_dropout_mask) = self.dropout_layer(input_char_embedding, input_char_embedding_mask)
            
            (input_char_conv,
                input_char_conv_mask) = self.conv_layer(input_char_dropout, input_char_dropout_mask)
            
            (input_char_pool,
                input_char_pool_mask) = self.pooling_layer(input_char_conv, input_char_conv_mask)
            
            input_char_feat = input_char_pool
            input_char_feat_mask = input_char_pool_mask
        
        return input_char_feat, input_char_feat_mask
