import numpy as np
import tensorflow as tf

from util.default_util import *
from util.sequence_labeling_util import *

__all__ = ["MaxPooling", "MaxPooling3D", "AveragePooling", "AveragePooling3D"]

class MaxPooling(object):
    """max pooling layer"""
    def __init__(self,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="max_pool"):
        """initialize max pooling layer"""
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call max pooling layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            output_mask = tf.squeeze(tf.reduce_max(input_mask, axis=-2, keepdims=True), axis=-2)
            output_pool = tf.reduce_max(input_data * input_mask + MIN_FLOAT * (1 - input_mask), axis=-2) * output_mask
            output_pool = output_pool + tf.reduce_max(input_data, axis=-2) * (1 - output_mask)
        
        return output_pool, output_mask

class MaxPooling3D(object):
    """max pooling layer"""
    def __init__(self,
                 window_size,
                 stride_size,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="max_pool_3d"):
        """initialize 3d max pooling layer"""
        self.window_size = window_size
        self.stride_size = stride_size
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.pooling_layer = tf.layers.MaxPooling3D(self.window_size, self.stride_size, "VALID")
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call 3d max pooling layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data_shape = tf.shape(input_data)
            input_mask_shape = tf.shape(input_mask)
            shape_size = len(input_data.get_shape().as_list())
            if shape_size > 5:
                input_pooling = tf.reshape(input_data, shape=tf.concat([[-1], input_data_shape[-4:]], axis=0))
                input_pooling_mask = tf.reshape(input_mask, shape=tf.concat([[-1], input_mask_shape[-4:]], axis=0))
            else:
                input_pooling = input_data
                input_pooling_mask = input_mask
            
            output_pooling = self.pooling_layer(input_pooling)
            output_mask = self.pooling_layer(input_pooling_mask)
            
            if shape_size > 5:
                output_pooling_shape = tf.shape(output_pooling)
                output_mask_shape = tf.shape(output_mask)
                output_pooling = tf.reshape(output_pooling,
                    shape=tf.concat([input_data_shape[:-4], output_pooling_shape[-4:]], axis=0))
                output_mask = tf.reshape(output_mask,
                    shape=tf.concat([input_mask_shape[:-4], output_mask_shape[-4:]], axis=0))
        
        return output_pooling, output_mask

class AveragePooling(object):
    """average pooling layer"""
    def __init__(self,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="avg_pool"):
        """initialize average pooling layer"""
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call average pooling layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_sum = tf.reduce_sum(input_data * input_mask, axis=-2)
            input_count = tf.count_nonzero(input_mask, axis=-2, dtype=tf.float32)
            output_mask = tf.squeeze(tf.reduce_max(input_mask, axis=-2, keepdims=True), axis=-2)
            output_pool = 1.0 * input_sum / (input_count - output_mask + 1.0)
        
        return output_pool, output_mask

class AveragePooling3D(object):
    """average pooling layer"""
    def __init__(self,
                 window_size,
                 stride_size,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="avg_pool_3d"):
        """initialize 3d average pooling layer"""
        self.window_size = window_size
        self.stride_size = stride_size
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.pooling_layer = tf.layers.AveragePooling3D(self.window_size, self.stride_size, "VALID")
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call 3d average pooling layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data_shape = tf.shape(input_data)
            input_mask_shape = tf.shape(input_mask)
            shape_size = len(input_data.get_shape().as_list())
            if shape_size > 5:
                input_pooling = tf.reshape(input_data, shape=tf.concat([[-1], input_data_shape[-4:]], axis=0))
                input_pooling_mask = tf.reshape(input_mask, shape=tf.concat([[-1], input_mask_shape[-4:]], axis=0))
            else:
                input_pooling = input_data
                input_pooling_mask = input_mask
            
            output_pooling = self.pooling_layer(input_pooling)
            output_mask = tf.cast(tf.greater_equal(self.pooling_layer(input_pooling_mask),
                tf.constant(0.0, shape=[], dtype=tf.float32)), dtype=tf.float32)
            
            if shape_size > 5:
                output_pooling_shape = tf.shape(output_pooling)
                output_mask_shape = tf.shape(output_mask)
                output_pooling = tf.reshape(output_pooling,
                    shape=tf.concat([input_data_shape[:-4], output_pooling_shape[-4:]], axis=0))
                output_mask = tf.reshape(output_mask,
                    shape=tf.concat([input_mask_shape[:-4], output_mask_shape[-4:]], axis=0))
        
        return output_pooling, output_mask
