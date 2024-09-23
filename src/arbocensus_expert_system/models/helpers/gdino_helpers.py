import numpy as np

def tensord_2_arraytuple(tensor_dict):
    labels = tensor_dict[0]['labels']
    boxes = tensor_dict[0]['boxes'].cpu().numpy().astype(int)
    scores = tensor_dict[0]['scores'].cpu().numpy()

    return labels, boxes, scores

def get_box_mask(image, box):
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[box[1]:box[3], box[0]:box[2]] = True
    return mask