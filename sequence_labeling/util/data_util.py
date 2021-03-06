import codecs
import collections
import os.path
import json

import numpy as np
import tensorflow as tf

from util.default_util import *

__all__ = ["DataPipeline", "create_online_pipeline", "create_dynamic_pipeline", "create_data_pipeline",
           "create_text_dataset", "create_label_dataset", "create_ext_dataset",
           "generate_word_feat", "generate_char_feat", "generate_label_feat",
           "create_embedding_file", "load_embedding_file", "convert_embedding",
           "create_vocab_file", "load_vocab_file", "process_vocab_table",
           "create_word_vocab", "create_char_vocab", "create_label_vocab",
           "load_tsv_data", "load_json_data", "load_sequence_data",
           "prepare_text_data", "prepare_label_data", "prepare_sequence_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "word_vocab_size", "char_vocab_size", "input_text_word", "input_text_char",
     "input_label", "input_ext", "input_text_word_mask", "input_text_char_mask", "input_label_mask", "input_ext_mask",
     "label_inverted_index", "input_text_placeholder", "input_word_placeholder", "input_char_placeholder",
     "input_label_placeholder", "input_ext_placeholder", "data_size_placeholder", "batch_size_placeholder"))):
    pass

def create_online_pipeline(external_index_enable,
                           word_vocab_size,
                           word_vocab_index,
                           word_max_size,
                           word_pad,
                           word_feat_enable,
                           char_vocab_size,
                           char_vocab_index,
                           char_max_size,
                           char_pad,
                           char_feat_enable,
                           label_inverted_index,
                           ext_max_size,
                           ext_embed_dim,
                           ext_feat_enable):
    """create online data pipeline for sequence labeling model"""
    input_text_placeholder = None
    input_word_placeholder = None
    input_char_placeholder = None
    input_text_word = None
    input_text_word_mask = None
    input_text_char = None
    input_text_char_mask = None
    input_ext = None
    input_ext_mask = None
    
    if external_index_enable == True:
        input_word_placeholder = tf.placeholder(shape=[None, None], dtype=tf.int32)
        input_char_placeholder = tf.placeholder(shape=[None, None, None], dtype=tf.int32)
        input_ext_placeholder = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
        if word_feat_enable == True:
            word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
            input_text_word = tf.expand_dims(input_word_placeholder[:,:word_max_size], axis=-1)
            input_text_word_mask = tf.cast(tf.not_equal(input_text_word, word_pad_id), dtype=tf.float32)
        
        if char_feat_enable == True:
            char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
            input_text_char = input_char_placeholder[:,:word_max_size,:char_max_size]
            input_text_char_mask = tf.cast(tf.not_equal(input_text_char, char_pad_id), dtype=tf.float32)
        
        if ext_feat_enable == True:
            ext_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
            input_ext = input_ext_placeholder[:,:ext_max_size,:ext_embed_dim]
            input_ext_mask = tf.expand_dims(input_word_placeholder[:,:ext_max_size], axis=-1)
            input_ext_mask = tf.cast(tf.not_equal(input_ext_mask, ext_pad_id), dtype=tf.float32)
    else:
        input_text_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        input_ext_placeholder = tf.placeholder(shape=[None, None, None], dtype=tf.float32)
        if word_feat_enable == True:
            word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
            input_text_word = tf.map_fn(lambda sent: generate_word_feat(sent,
                word_vocab_index, word_max_size, word_pad), input_text_placeholder, dtype=tf.int32)
            input_text_word_mask = tf.cast(tf.not_equal(input_text_word, word_pad_id), dtype=tf.float32)
        
        if char_feat_enable == True:
            char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
            input_text_char = tf.map_fn(lambda sent: generate_char_feat(sent,
                word_max_size, char_vocab_index, char_max_size, char_pad), input_text_placeholder, dtype=tf.int32)
            input_text_char_mask = tf.cast(tf.not_equal(input_text_char, char_pad_id), dtype=tf.float32)
        
        if ext_feat_enable == True:
            ext_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
            input_ext = input_ext_placeholder[:,:ext_max_size,:ext_embed_dim]
            input_ext_mask = tf.map_fn(lambda sent: generate_word_feat(sent,
                word_vocab_index, word_max_size, word_pad), input_text_placeholder, dtype=tf.int32)
            input_ext_mask = tf.cast(tf.not_equal(input_ext_mask, ext_pad_id), dtype=tf.float32)
    
    return DataPipeline(initializer=None, word_vocab_size=word_vocab_size, char_vocab_size=char_vocab_size,
        input_text_word=input_text_word, input_text_char=input_text_char, input_label=None, input_ext=input_ext,
        input_text_word_mask=input_text_word_mask, input_text_char_mask=input_text_char_mask, input_label_mask=None,
        input_ext_mask=input_ext_mask, label_inverted_index=label_inverted_index, input_text_placeholder=input_text_placeholder,
        input_word_placeholder=input_word_placeholder, input_char_placeholder=input_char_placeholder,
        input_label_placeholder=None, input_ext_placeholder=None, data_size_placeholder=None, batch_size_placeholder=None)

def create_dynamic_pipeline(input_text_word_dataset,
                            input_text_char_dataset,
                            input_label_dataset,
                            input_ext_dataset,
                            word_vocab_size,
                            word_vocab_index,
                            word_pad,
                            word_feat_enable,
                            char_vocab_size,
                            char_vocab_index,
                            char_pad,
                            char_feat_enable,
                            label_vocab_index,
                            label_inverted_index,
                            label_pad,
                            ext_feat_enable,
                            random_seed,
                            enable_shuffle,
                            buffer_size,
                            input_text_placeholder,
                            input_label_placeholder,
                            input_ext_placeholder,
                            data_size_placeholder,
                            batch_size_placeholder):
    """create dynamic data pipeline for sequence labeling model"""
    if word_feat_enable == True:
        word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        word_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
        input_text_word_dataset = tf.data.Dataset.from_tensors(
            tf.constant(0, shape=[1,1], dtype=tf.float32)).repeat(data_size)
    
    if char_feat_enable == True:
        char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
    else:
        char_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
        input_text_char_dataset = tf.data.Dataset.from_tensors(
            tf.constant(0, shape=[1,1], dtype=tf.float32)).repeat(data_size)
    
    if ext_feat_enable == True:
        ext_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        ext_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
        input_ext_dataset = tf.data.Dataset.from_tensors(
            tf.constant(0, shape=[1,1], dtype=tf.float32)).repeat(data_size)
    
    dataset = tf.data.Dataset.zip((input_text_word_dataset,
        input_text_char_dataset, input_label_dataset, input_ext_dataset))
    
    if enable_shuffle == True:
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.batch(batch_size=batch_size_placeholder)
    dataset = dataset.prefetch(buffer_size=1)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if word_feat_enable == True:
        input_text_word = tf.cast(batch_data[0], dtype=tf.int32)
        input_text_word_mask = tf.cast(tf.not_equal(input_text_word, word_pad_id), dtype=tf.float32)
    else:
        input_text_word = None
        input_text_word_mask = None
    
    if char_feat_enable == True:
        input_text_char = tf.cast(batch_data[1], dtype=tf.int32)
        input_text_char_mask = tf.cast(tf.not_equal(input_text_char, char_pad_id), dtype=tf.float32)
    else:
        input_text_char = None
        input_text_char_mask = None
    
    label_pad_id = tf.cast(label_vocab_index.lookup(tf.constant(label_pad)), dtype=tf.float32)
    input_label = tf.cast(batch_data[2], dtype=tf.float32)
    input_label_mask = tf.cast(tf.not_equal(input_label, label_pad_id), dtype=tf.float32)
    
    if ext_feat_enable == True:
        input_ext = tf.cast(batch_data[3], dtype=tf.float32)
        input_ext_mask = tf.cast(batch_data[0], dtype=tf.int32)
        input_ext_mask = tf.cast(tf.not_equal(input_ext_mask, ext_pad_id), dtype=tf.float32)
    else:
        input_ext = None
        input_ext_mask = None
    
    return DataPipeline(initializer=iterator.initializer, word_vocab_size=word_vocab_size, char_vocab_size=char_vocab_size,
        input_text_word=input_text_word, input_text_char=input_text_char, input_label=input_label, input_ext=input_ext,
        input_text_word_mask=input_text_word_mask, input_text_char_mask=input_text_char_mask,
        input_label_mask=input_label_mask, input_ext_mask=input_ext_mask, label_inverted_index=label_inverted_index,
        input_text_placeholder=input_text_placeholder, input_word_placeholder=None, input_char_placeholder=None,
        input_label_placeholder=input_label_placeholder, input_ext_placeholder=input_ext_placeholder,
        data_size_placeholder=data_size_placeholder, batch_size_placeholder=batch_size_placeholder)

def create_data_pipeline(input_text_word_dataset,
                         input_text_char_dataset,
                         input_label_dataset,
                         input_ext_dataset,
                         word_vocab_size,
                         word_vocab_index,
                         word_pad,
                         word_feat_enable,
                         char_vocab_size,
                         char_vocab_index,
                         char_pad,
                         char_feat_enable,
                         label_vocab_index,
                         label_inverted_index,
                         label_pad,
                         ext_feat_enable,
                         random_seed,
                         enable_shuffle,
                         buffer_size,
                         data_size,
                         batch_size):
    """create data pipeline for sequence labeling model"""
    if word_feat_enable == True:
        word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        word_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
        input_text_word_dataset = tf.data.Dataset.from_tensors(
            tf.constant(0, shape=[1,1], dtype=tf.float32)).repeat(data_size)
    
    if char_feat_enable == True:
        char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
    else:
        char_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
        input_text_char_dataset = tf.data.Dataset.from_tensors(
            tf.constant(0, shape=[1,1], dtype=tf.float32)).repeat(data_size)
    
    if ext_feat_enable == True:
        ext_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        ext_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
        input_ext_dataset = tf.data.Dataset.from_tensors(
            tf.constant(0, shape=[1,1], dtype=tf.float32)).repeat(data_size)
    
    dataset = tf.data.Dataset.zip((input_text_word_dataset,
        input_text_char_dataset, input_label_dataset, input_ext_dataset))
    
    if enable_shuffle == True:
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if word_feat_enable == True:
        input_text_word = tf.cast(batch_data[0], dtype=tf.int32)
        input_text_word_mask = tf.cast(tf.not_equal(input_text_word, word_pad_id), dtype=tf.float32)
    else:
        input_text_word = None
        input_text_word_mask = None
    
    if char_feat_enable == True:
        input_text_char = tf.cast(batch_data[1], dtype=tf.int32)
        input_text_char_mask = tf.cast(tf.not_equal(input_text_char, char_pad_id), dtype=tf.float32)
    else:
        input_text_char = None
        input_text_char_mask = None
    
    label_pad_id = tf.cast(label_vocab_index.lookup(tf.constant(label_pad)), dtype=tf.float32)
    input_label = tf.cast(batch_data[2], dtype=tf.float32)
    input_label_mask = tf.cast(tf.not_equal(input_label, label_pad_id), dtype=tf.float32)
    
    if ext_feat_enable == True:
        input_ext = tf.cast(batch_data[3], dtype=tf.float32)
        input_ext_mask = tf.cast(batch_data[0], dtype=tf.int32)
        input_ext_mask = tf.cast(tf.not_equal(input_ext_mask, ext_pad_id), dtype=tf.float32)
    else:
        input_ext = None
        input_ext_mask = None
    
    return DataPipeline(initializer=iterator.initializer, word_vocab_size=word_vocab_size, char_vocab_size=char_vocab_size,
        input_text_word=input_text_word, input_text_char=input_text_char, input_label=input_label, input_ext=input_ext,
        input_text_word_mask=input_text_word_mask, input_text_char_mask=input_text_char_mask,
        input_label_mask=input_label_mask, input_ext_mask=input_ext_mask, label_inverted_index=label_inverted_index,
        input_text_placeholder=None, input_word_placeholder=None, input_char_placeholder=None, 
        input_label_placeholder=None, input_ext_placeholder=None, data_size_placeholder=None, batch_size_placeholder=None)

def create_text_dataset(input_dataset,
                        word_vocab_index,
                        word_max_size,
                        word_pad,
                        word_feat_enable,
                        char_vocab_index,
                        char_max_size,
                        char_pad,
                        char_feat_enable,
                        num_parallel):
    """create word/char-level dataset for input data"""
    word_dataset = None
    if word_feat_enable == True:
        word_dataset = input_dataset.map(lambda sent: generate_word_feat(sent,
            word_vocab_index, word_max_size, word_pad), num_parallel_calls=num_parallel)
    
    char_dataset = None
    if char_feat_enable == True:
        char_dataset = input_dataset.map(lambda sent: generate_char_feat(sent,
            word_max_size, char_vocab_index, char_max_size, char_pad), num_parallel_calls=num_parallel)
    
    return word_dataset, char_dataset

def create_label_dataset(input_dataset,
                         label_vocab_index,
                         label_max_size,
                         label_pad,
                         num_parallel):
    """create label dataset for input data"""
    label_dataset = input_dataset.map(lambda sent: generate_label_feat(sent,
        label_vocab_index, label_max_size, label_pad), num_parallel_calls=num_parallel)
    
    return label_dataset

def create_ext_dataset(input_dataset,
                       ext_max_size,
                       ext_embed_dim,
                       ext_pad,
                       ext_feat_enable,
                       num_parallel):
    """create extended dataset for input data"""
    ext_dataset = None
    if ext_feat_enable == True:
        ext_dataset = input_dataset.map(lambda ext: generate_ext_feat(ext,
            ext_max_size, ext_embed_dim, ext_pad), num_parallel_calls=num_parallel)
    
    return ext_dataset

def generate_word_feat(sentence,
                       word_vocab_index,
                       word_max_size,
                       word_pad):
    """process words for sentence"""
    sentence_words = tf.string_split([sentence], delimiter=' ').values
    sentence_words = tf.concat([sentence_words[:word_max_size],
        tf.constant(word_pad, shape=[word_max_size])], axis=0)
    sentence_words = tf.reshape(sentence_words[:word_max_size], shape=[word_max_size])
    sentence_words = tf.cast(word_vocab_index.lookup(sentence_words), dtype=tf.float32)
    sentence_words = tf.expand_dims(sentence_words, axis=-1)
    
    return sentence_words

def generate_char_feat(sentence,
                       word_max_size,
                       char_vocab_index,
                       char_max_size,
                       char_pad):
    """generate characters for sentence"""
    def word_to_char(word):
        """process characters for word"""
        word_chars = tf.string_split([word], delimiter='').values
        word_chars = tf.concat([word_chars[:char_max_size],
            tf.constant(char_pad, shape=[char_max_size])], axis=0)
        word_chars = tf.reshape(word_chars[:char_max_size], shape=[char_max_size])
        
        return word_chars
    
    sentence_words = tf.string_split([sentence], delimiter=' ').values
    sentence_words = tf.concat([sentence_words[:word_max_size],
        tf.constant(char_pad, shape=[word_max_size])], axis=0)
    sentence_words = tf.reshape(sentence_words[:word_max_size], shape=[word_max_size])
    sentence_chars = tf.map_fn(word_to_char, sentence_words)
    sentence_chars = tf.cast(char_vocab_index.lookup(sentence_chars), dtype=tf.float32)
    
    return sentence_chars

def generate_label_feat(sentence,
                        label_vocab_index,
                        label_max_size,
                        label_pad):
    """process labels for sentence"""
    sentence_labels = tf.string_split([sentence], delimiter=' ').values
    sentence_labels = tf.concat([sentence_labels[:label_max_size],
        tf.constant(label_pad, shape=[label_max_size])], axis=0)
    sentence_labels = tf.reshape(sentence_labels[:label_max_size], shape=[label_max_size])
    sentence_labels = tf.cast(label_vocab_index.lookup(sentence_labels), dtype=tf.float32)
    sentence_labels = tf.expand_dims(sentence_labels, axis=-1)
    
    return sentence_labels

def generate_ext_feat(ext_data,
                      ext_max_size,
                      ext_embed_dim,
                      ext_pad):
    """process words for sentence"""
    ext_feat = tf.concat([ext_data[:ext_max_size,:ext_embed_dim],
        tf.constant(ext_pad, shape=[ext_max_size,ext_embed_dim])], axis=0)
    ext_feat = tf.reshape(ext_feat[:ext_max_size,:ext_embed_dim], shape=[ext_max_size,ext_embed_dim])
    ext_feat = tf.cast(ext_feat, dtype=tf.float32)
    
    return ext_feat

def create_embedding_file(embedding_file,
                          embedding_table):
    """create embedding file based on embedding table"""
    embedding_dir = os.path.dirname(embedding_file)
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)
    
    if not os.path.exists(embedding_file):
        with codecs.getwriter("utf-8")(open(embedding_file, "wb")) as file:
            for vocab in embedding_table.keys():
                embed = embedding_table[vocab]
                embed_str = " ".join(map(str, embed))
                file.write("{0} {1}\n".format(vocab, embed_str))

def load_embedding_file(embedding_file,
                        embedding_size,
                        unk,
                        pad):
    """load pre-train embeddings from embedding file"""
    if os.path.exists(embedding_file):
        with codecs.getreader("utf-8")(open(embedding_file, "rb")) as file:
            embedding = {}
            for line in file:
                items = line.strip().split(' ')
                if len(items) != embedding_size + 1:
                    continue
                word = items[0]
                vector = [float(x) for x in items[1:]]
                if word not in embedding:
                    embedding[word] = vector
            
            if unk not in embedding:
                embedding[unk] = np.random.rand(embedding_size)
            if pad not in embedding:
                embedding[pad] = np.random.rand(embedding_size)
            
            return embedding
    else:
        raise FileNotFoundError("embedding file not found")

def convert_embedding(embedding_lookup):
    if embedding_lookup is not None:
        embedding = [v for k,v in embedding_lookup.items()]
    else:
        embedding = None
    
    return embedding

def create_vocab_file(vocab_file,
                      vocab_table):
    """create vocab file based on vocab table"""
    vocab_dir = os.path.dirname(vocab_file)
    if not os.path.exists(vocab_dir):
        os.mkdir(vocab_dir)
    
    if not os.path.exists(vocab_file):
        with codecs.getwriter("utf-8")(open(vocab_file, "wb")) as file:
            for vocab in vocab_table:
                file.write("{0}\n".format(vocab))

def load_vocab_file(vocab_file):
    """load vocab data from vocab file"""
    if os.path.exists(vocab_file):
        with codecs.getreader("utf-8")(open(vocab_file, "rb")) as file:
            vocab = {}
            for line in file:
                items = line.strip().split('\t')
                
                item_size = len(items)
                if item_size > 1:
                    vocab[items[0]] = int(items[1])
                elif item_size > 0:
                    vocab[items[0]] = MAX_INT
            
            return vocab
    else:
        raise FileNotFoundError("vocab file not found")

def process_vocab_table(vocab,
                        vocab_size,
                        vocab_threshold,
                        vocab_lookup,
                        unk,
                        pad):
    """process vocab table"""
    if unk == pad:
        raise ValueError("unknown vocab {0} can not be the same as padding vocab {1}".format(unk, pad))
    
    if unk in vocab:
        del vocab[unk]
    if pad in vocab:
        del vocab[pad]
    
    vocab = { k: vocab[k] for k in vocab.keys() if vocab[k] >= vocab_threshold }
    if vocab_lookup is not None:
        vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
    
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    vocab_table = [unk] + sorted_vocab[:vocab_size-2] + [pad] if unk else sorted_vocab[:vocab_size-1] + [pad]
    
    vocab_size = len(vocab_table)
    default_vocab = vocab_table[0]
    
    vocab_index = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(vocab_table), default_value=0)
    vocab_inverted_index = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=tf.constant(vocab_table), default_value=default_vocab)
    
    return vocab_table, vocab_size, vocab_index, vocab_inverted_index

def create_word_vocab(input_data):
    """create word vocab from input data"""
    word_vocab = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            if word not in word_vocab:
                word_vocab[word] = 1
            else:
                word_vocab[word] += 1
    
    return word_vocab

def create_char_vocab(input_data):
    """create char vocab from input data"""
    char_vocab = {}
    for sentence in input_data:
        words = sentence.strip().split(' ')
        for word in words:
            chars = list(word)
            for ch in chars:
                if ch not in char_vocab:
                    char_vocab[ch] = 1
                else:
                    char_vocab[ch] += 1
    
    return char_vocab

def create_label_vocab(input_data):
    """create label vocab from input data"""
    label_vocab = {}
    for sentence in input_data:
        labels = sentence.strip().split(' ')
        for label in labels:
            if label not in label_vocab:
                label_vocab[label] = 1
            else:
                label_vocab[label] += 1
    
    return label_vocab

def load_tsv_data(input_file):
    """load data from tsv file"""
    if os.path.exists(input_file):
        with codecs.getreader("utf-8")(open(input_file, "rb")) as file:
            input_data = []
            item_separator = "\t"
            for line in file:
                items = line.strip().split(item_separator)
                if len(items) < 3:
                    continue
                
                sample_id = items[0]
                text = items[1]
                label = items[2]
                
                if len(items) > 3:
                    ext_data = json.loads(items[3])
                else:
                    ext_data = []
                
                input_data.append({
                    "id": sample_id,
                    "text": text,
                    "label": label,
                    "extended": ext_data
                })
            
            return input_data
    else:
        raise FileNotFoundError("input file not found")

def load_json_data(input_file):
    """load data from json file"""
    if os.path.exists(input_file):
        with codecs.getreader("utf-8")(open(input_file, "rb") ) as file:
            input_data = json.load(file)
            return input_data
    else:
        raise FileNotFoundError("input file not found")

def load_sequence_data(input_file,
                       file_type):
    """load sequence data from input file"""
    if file_type == "tsv":
        input_data = load_tsv_data(input_file)
    elif file_type == "json":
        input_data = load_json_data(input_file)
    else:
        raise ValueError("can not load data from unsupported file type {0}".format(file_type))
    
    return input_data

def prepare_text_data(logger,
                      input_data,
                      word_vocab_file,
                      word_vocab_size,
                      word_vocab_threshold,
                      word_embed_dim,
                      word_embed_file,
                      full_word_embed_file,
                      word_unk,
                      word_pad,
                      word_feat_enable,
                      pretrain_word_embed,
                      char_vocab_file,
                      char_vocab_size,
                      char_vocab_threshold,
                      char_unk,
                      char_pad,
                      char_feat_enable):
    """prepare text data"""    
    word_embed_data = None
    if pretrain_word_embed == True:
        if os.path.exists(word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(word_embed_file))
            word_embed_data = load_embedding_file(word_embed_file, word_embed_dim, word_unk, word_pad)
        elif os.path.exists(full_word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(full_word_embed_file))
            word_embed_data = load_embedding_file(full_word_embed_file, word_embed_dim, word_unk, word_pad)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_embed_file, full_word_embed_file))
        
        word_embed_size = len(word_embed_data) if word_embed_data is not None else 0
        logger.log_print("# word embedding table has {0} words".format(word_embed_size))
    
    word_vocab = None
    word_vocab_index = None
    word_vocab_inverted_index = None
    if os.path.exists(word_vocab_file):
        logger.log_print("# loading word vocab table from {0}".format(word_vocab_file))
        word_vocab = load_vocab_file(word_vocab_file)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad)
    elif input_data is not None:
        logger.log_print("# creating word vocab table from input data")
        word_vocab = create_word_vocab(input_data)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad)
        logger.log_print("# creating word vocab file {0}".format(word_vocab_file))
        create_vocab_file(word_vocab_file, word_vocab_table)
    else:
        raise ValueError("{0} or input data must be provided".format(word_vocab_file))
    
    logger.log_print("# word vocab table has {0} words".format(word_vocab_size))
    
    char_vocab = None
    char_vocab_index = None
    char_vocab_inverted_index = None
    if char_feat_enable is True:
        if os.path.exists(char_vocab_file):
            logger.log_print("# loading char vocab table from {0}".format(char_vocab_file))
            char_vocab = load_vocab_file(char_vocab_file)
            (_, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad)
        elif input_data is not None:
            logger.log_print("# creating char vocab table from input data")
            char_vocab = create_char_vocab(input_data)
            (char_vocab_table, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad)
            logger.log_print("# creating char vocab file {0}".format(char_vocab_file))
            create_vocab_file(char_vocab_file, char_vocab_table)
        else:
            raise ValueError("{0} or input data must be provided".format(char_vocab_file))

        logger.log_print("# char vocab table has {0} chars".format(char_vocab_size))
    
    if word_embed_data is not None and word_vocab_table is not None:
        word_embed_data = { k: word_embed_data[k] for k in word_vocab_table if k in word_embed_data }
        logger.log_print("# word embedding table has {0} words after filtering".format(len(word_embed_data)))
        if not os.path.exists(word_embed_file):
            logger.log_print("# creating word embedding file {0}".format(word_embed_file))
            create_embedding_file(word_embed_file, word_embed_data)
        
        word_embed_data = convert_embedding(word_embed_data)
    
    return (word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index)

def prepare_label_data(logger,
                       input_data,
                       label_vocab_file,
                       label_vocab_size,
                       label_unk,
                       label_pad):
    """prepare label data"""
    label_vocab = None
    label_vocab_index = None
    label_vocab_inverted_index = None
    if os.path.exists(label_vocab_file):
        logger.log_print("# loading label vocab table from {0}".format(label_vocab_file))
        label_vocab = load_vocab_file(label_vocab_file)
        (label_vocab_table, label_vocab_size, label_vocab_index,
            label_vocab_inverted_index) = process_vocab_table(label_vocab, label_vocab_size,
            0, None, label_unk, label_pad)
    elif input_data is not None:
        logger.log_print("# creating label vocab table from input data")
        label_vocab = create_label_vocab(input_data)
        (label_vocab_table, label_vocab_size, label_vocab_index,
            label_vocab_inverted_index) = process_vocab_table(label_vocab, label_vocab_size,
            0, None, label_unk, label_pad)
        logger.log_print("# creating label vocab file {0}".format(label_vocab_file))
        create_vocab_file(label_vocab_file, label_vocab_table)
    else:
        raise ValueError("{0} or input data must be provided".format(label_vocab_file))
    
    logger.log_print("# label vocab table has {0} words".format(label_vocab_size))
    
    return label_vocab_size, label_vocab_index, label_vocab_inverted_index

def prepare_sequence_data(logger,
                          input_sequence_file,
                          input_file_type,
                          word_vocab_file,
                          word_vocab_size,
                          word_vocab_threshold,
                          word_embed_dim,
                          word_embed_file,
                          full_word_embed_file,
                          word_unk,
                          word_pad,
                          word_feat_enable,
                          pretrain_word_embed,
                          char_vocab_file,
                          char_vocab_size,
                          char_vocab_threshold,
                          char_unk,
                          char_pad,
                          char_feat_enable,
                          label_vocab_file,
                          label_vocab_size,
                          label_unk,
                          label_pad):
    """prepare sequence data"""
    logger.log_print("# loading input sequence data from {0}".format(input_sequence_file))
    input_sequence_data = load_sequence_data(input_sequence_file, input_file_type)

    input_sequence_size = len(input_sequence_data)
    logger.log_print("# input sequence data has {0} lines".format(input_sequence_size))
    
    input_text_data = [sequence_data["text"] for sequence_data in input_sequence_data]
    input_label_data = [sequence_data["label"] for sequence_data in input_sequence_data]
    input_ext_data = [sequence_data["extended"] if "extended" in sequence_data else [] for sequence_data in input_sequence_data]
    
    (word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_text_data(logger, input_text_data,
            word_vocab_file, word_vocab_size, word_vocab_threshold, word_embed_dim, word_embed_file,
            full_word_embed_file, word_unk, word_pad, word_feat_enable, pretrain_word_embed,
            char_vocab_file, char_vocab_size, char_vocab_threshold, char_unk, char_pad, char_feat_enable)
    
    label_vocab_size, label_vocab_index, label_vocab_inverted_index = prepare_label_data(logger, input_label_data,
        label_vocab_file, label_vocab_size, label_unk, label_pad)
    
    return (input_sequence_data, input_text_data, input_label_data, input_ext_data,
        word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index,
        label_vocab_size, label_vocab_index, label_vocab_inverted_index)
