from arbocensus_expert_system.models.mask_extractor_sam import MaskExtractorSAM
from arbocensus_expert_system.models.monocular_depth_dam import MonocularDepthDAM
from arbocensus_expert_system.models.gdino_object_detector import GDinoObjectDetector
from arbocensus_expert_system.models.tree_classifier_resnet import TreeClassifierResnet
from arbocensus_expert_system.models.tree_image import TreeImage
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

    