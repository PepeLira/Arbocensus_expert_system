from models.mask_extractor_sam import MaskExtractorSAM
from models.monocular_depth_dam import MonocularDepthDAM
from models.gdino_object_detector import GDinoObjectDetector
from abc import ABC, abstractmethod
from models.tree_image import TreeImage

class TreeSegmentationStrategy(ABC):
    @abstractmethod
    def __init__(self, mask_extractor: MaskExtractorSAM, depth_mapper: MonocularDepthDAM, object_classifier: GDinoObjectDetector):
        pass
    
    @abstractmethod
    def segment(self, tree_image, plot=False) -> TreeImage:
        pass

    