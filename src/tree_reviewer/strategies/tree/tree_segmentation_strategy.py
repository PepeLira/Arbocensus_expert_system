from tree_reviewer.models.mask_extractor_sam import MaskExtractorSAM
from tree_reviewer.models.monocular_depth_dam import MonocularDepthDAM
from tree_reviewer.models.gdino_object_detector import GDinoObjectDetector
from tree_reviewer.models.tree_classifier_resnet import TreeClassifierResnet
from tree_reviewer.models.tree_image import TreeImage
from abc import ABC, abstractmethod

class TreeSegmentationStrategy(ABC):
    @abstractmethod
    def __init__(self, mask_extractor: MaskExtractorSAM, 
                 depth_mapper: MonocularDepthDAM, 
                 object_classifier: GDinoObjectDetector,
                 species_classifier: TreeClassifierResnet):
        pass
    
    @abstractmethod
    def segment(self, tree_image, plot=False) -> TreeImage:
        pass

    