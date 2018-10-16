import numpy as np
import tensorflow as tf

from external.precision_recall import *

__all__ = ["evaluate_from_data"]

def _precision(pred_data, label_data):
    """Precision"""
    precision, _ = get_precision_recall(pred_data, label_data)
    return precision

def _recall(pred_data, label_data):
    """Recall"""
    _, recall = get_precision_recall(pred_data, label_data)
    return recall

def _f1_score(pred_data, label_data):
    """F1 score"""
    precision, recall = get_precision_recall(pred_data, label_data)
    f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return f1_score

def evaluate_from_data(pred_data, label_data, metric):
    """compute evaluation score based on selected metric"""
    if len(pred_data) == 0 or len(label_data) == 0:
        return 0.0
    
    if metric == "precision":
        eval_score = _precision(pred_data, label_data)
    elif metric == "recall":
        eval_score = _recall(pred_data, label_data)
    elif metric == "f1":
        eval_score = _f1_score(pred_data, label_data)
    else:
        raise ValueError("unsupported metric {0}".format(metric))
    
    return eval_score
