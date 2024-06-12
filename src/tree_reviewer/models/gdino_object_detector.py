from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import numpy as np
from .helpers.gdino_helpers import tensord_2_arraytuple, get_box_mask
import pdb

class GDinoObjectDetector:
    def __init__(self):
        model_id = "IDEA-Research/grounding-dino-base"
        self.device = self.get_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def detect(self, image, text_prompt):
        inputs = self.processor(images=image, text=text_prompt, 
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.2,
            text_threshold=0.3,
            target_sizes=[image.shape[:-1]]
        )
        labels, boxes, scores = tensord_2_arraytuple(results)

        # print(labels, boxes, scores)
        return labels, boxes, scores
    
    def classify_object(self, image, text_prompt, target_words=["tree", "palm"]):
        """Gets the best target classification score"""
        labels, boxes, scores = self.detect(image(), text_prompt)
        target_word_indexes = self.get_target_word_indexes(labels, target_words)
        best_boxes_indexes = self.get_best_boxes_indexes(image, boxes)
        target_word_indexes = [i for i in target_word_indexes if i in best_boxes_indexes]

        if len(target_word_indexes) == 0:
            return 0, None 
        else:
            best_score_index = np.argmax(scores[target_word_indexes])
            return scores[best_score_index], boxes[best_score_index]
        
    def get_target_word_indexes(self, labels, target_words=["tree", "palm"]):
        target_word_indexes = []
        for i, label in enumerate(labels):
            for l in label.split():
                if l in target_words:
                    target_word_indexes.append(i)
        return target_word_indexes
    
    def get_best_boxes_indexes(self, image, boxes, threshold=0.2):
        object_mask = image.get_mask()
        best_boxes_indexes = []
        if object_mask is None:
            for i, box in enumerate(boxes):
                best_boxes_indexes.append(i)
        else:
            object_mask = object_mask.astype(bool)
            for i, box in enumerate(boxes):
                box_mask = get_box_mask(image(), box)
                intersection = np.logical_and(object_mask, box_mask)
                union = np.logical_or(object_mask, box_mask)
                iou = np.sum(intersection) / np.sum(union)
                if iou > threshold:
                    best_boxes_indexes.append(i)

        return best_boxes_indexes



    def get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return "CPU"