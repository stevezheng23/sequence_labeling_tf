import numpy as np
import tensorflow as tf

from external.accuracy import *
from external.precision_recall import *

__all__ = ["evaluate_from_data"]

def _accuracy(pred_data, label_data, invalid_labels):
    """Accuracy"""
    accuracy = get_accuracy(pred_data, label_data, invalid_labels)
    return accuracy

def _precision(pred_data, label_data, invalid_labels):
    """Precision"""
    precision, _ = get_precision_recall(pred_data, label_data, invalid_labels)
    return precision

def _recall(pred_data, label_data, invalid_labels):
    """Recall"""
    _, recall = get_precision_recall(pred_data, label_data, invalid_labels)
    return recall

def _f1_score(pred_data, label_data, invalid_labels):
    """F1 score"""
    precision, recall = get_precision_recall(pred_data, label_data, invalid_labels)
    f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return f1_score

def evaluate_from_data(pred_data, label_data, metric, invalid_labels):
    """compute evaluation score based on selected metric"""
    if len(pred_data) == 0 or len(label_data) == 0:
        return 0.0
    
    if metric == "accuracy":
        eval_score = _accuracy(pred_data, label_data, invalid_labels)
    elif metric == "precision":
        eval_score = _precision(pred_data, label_data, invalid_labels)
    elif metric == "recall":
        eval_score = _recall(pred_data, label_data, invalid_labels)
    elif metric == "f1":
        eval_score = _f1_score(pred_data, label_data, invalid_labels)
    else:
        raise ValueError("unsupported metric {0}".format(metric))
    
    return eval_score
