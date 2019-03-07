import numpy as np

__all__ = ["get_accuracy"]

def get_accuracy(predict_data,
                 label_data,
                 invalid_labels):
    if len(predict_data) != len(label_data):
        raise ValueError('# predict and # label must be the same')
    
    sample_data = list(zip(predict_data, label_data))
    if len(sample_data) == 0:
        raise ValueError('# sample must be more 0')
    
    sample_items = []
    for predict_items, label_items in sample_data:
        if len(predict_items) != len(label_items):
            raise ValueError('predict length must be equal to label length')
        
        sample_items.extend(list(zip(predict_items, label_items)))
    
    label_lookup = set(invalid_labels) if invalid_labels is not None else { "P" }
    correct, total = 0, 0
    for predict_item, label_item in sample_items:
        if predict_item in label_lookup or label_item in label_lookup:
            continue
        
        if predict_item == label_item:
            correct += 1
        
        total += 1
            
    accuracy = 0.0 if total == 0 else float(correct) / float(total)
    
    return accuracy
