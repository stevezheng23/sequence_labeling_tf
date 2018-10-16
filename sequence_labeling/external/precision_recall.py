import numpy as np

__all__ = ["get_precision_recall"]

def get_precision_recall(predict_data,
                         label_data):
    if len(predict_data) != len(label_data):
        raise ValueError('# predict and # label must be the same')
    
    sample_data = list(zip(predict_data, label_data))
    if len(sample_data) == 0:
        raise ValueError('# sample must be more 0')
    
    sample_items = []
    for predict_items, label_items in sample_data:
        if len(predict_items) != len(label_items):
            continue
        
        sample_items.extend(list(zip(predict_items, label_items)))
    
    label_lookup = { "O", "P" }
    tp, tp_tn, tp_fn = 0, 0, 0
    for predict_item, label_item in sample_items:
        if predict_item not in label_lookup:
            tp_tn += 1
        
        if label_item not in label_lookup:
            tp_fn += 1
        
        if predict_item in label_lookup or label_item in label_lookup:
            continue
        
        if predict_item == label_item:
            tp += 1
            
    precision = 0.0 if tp_tn == 0 else float(tp) / float(tp_tn)
    recall = 0.0 if tp_fn == 0 else float(tp) / float(tp_fn)
    
    return precision, recall
