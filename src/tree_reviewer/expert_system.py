print("Importing Dependencies...")
import os
import json
from tree_reviewer.image_loader import ImageLoader
from tree_reviewer.tree_reviewer import TreeReviewer
from tree_reviewer.models.tree_metrics import TreeMetrics
from tree_reviewer.models.mask_extractor_sam import MaskExtractorSAM
from tree_reviewer.models.monocular_depth_dam import MonocularDepthDAM
from tree_reviewer.models.gdino_object_detector import GDinoObjectDetector
from tree_reviewer.models.tree_classifier_resnet import TreeClassifierResnet
from tree_reviewer.strategies.tree.first_tree_segmenter import FirstTreeSegmenter
from tree_reviewer.strategies.card.card_segmentation_strategy import CardSegmentationStrategy
from tree_reviewer.config import get_env

# Ignore warnings remove for debugging
import warnings
warnings.filterwarnings("ignore")

import pdb

class ExpertSystem:
    def __init__(self, image_folder, output_path=os.getcwd(), with_masks=False, chunk_size=1, start_chunk=0):
        self.image_loader = ImageLoader(
            image_folder, 
            chunk_size, 
            start_chunk)
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.start_chunk = start_chunk
        self.with_masks = with_masks
        self.chunks_number = len(self.image_loader.data_chunks)
        self.init_models()
        self.select_segmentation_strategy()
        self.tree_reviewer = TreeReviewer(
            self.tree_segmentation_strategy, 
            self.card_segmentation_strategy)

    def init_models(self):
        print("Initializing Models...")
        self.mask_extractor = MaskExtractorSAM()
        self.depth_dam = MonocularDepthDAM()
        self.object_detector = GDinoObjectDetector()
        self.species_classifier = TreeClassifierResnet()

    def select_segmentation_strategy(self):
        self.tree_segmentation_strategy = FirstTreeSegmenter(
            self.mask_extractor, 
            self.depth_dam, 
            self.object_detector,
            self.species_classifier)
        self.card_segmentation_strategy = CardSegmentationStrategy(
            self.mask_extractor, 
            self.object_detector)
        
    def bulk_review(self, json_file_name=None):
        for json_file, tree_image, json_file_name in self.json_chunks(json_file_name):
            try:
                tree_metrics = self.tree_reviewer.review_tree(tree_image)
                if self.with_masks:
                    self.tree_reviewer.save_tree_figure(tree_metrics.tree_image, self.output_path)
                data = tree_metrics.get_data() 

                json.dump(data, json_file, indent=4, ensure_ascii=False)
                print(f"{self.image_loader.count}/{len(self.image_loader.current_chunk)} images processed.")

            except Exception as e:
                print(f"Error processing image: {tree_image.file}")
                tree_metrics = TreeMetrics(tree_image)
                tree_metrics.set_blank_data()
                data = tree_metrics.get_data()
                json.dump(data, json_file, indent=4, ensure_ascii=False)

    def json_chunks(self, json_file_name=None):
        for i in range(self.start_chunk, self.chunks_number):
            if json_file_name is None:
                json_file_name = self.image_loader.current_chunk_name() + str(i) + '.json'
            json_file_path = os.path.join(self.output_path, json_file_name) 

            # Create or overwrite the JSON file
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json_file.write('{"images": [')

            for tree_image in self.image_loader.load_image():
                with open(json_file_path, 'a', encoding='utf-8') as json_file:
                    yield json_file, tree_image, json_file_path
                    json_file.write(',\n')
            
            self.remove_last_comma(json_file_path)

            with open(json_file_path, 'a', encoding='utf-8') as json_file:
                json_file.write('\n]}')  # Closing the images array and JSON object.
            
            self.image_loader.next_chunk()

    def remove_last_comma(self, json_file_path):
        with open(json_file_path, 'rb+') as json_file:
            json_file.seek(-3, os.SEEK_END)
            json_file.truncate()

if __name__ == "__main__":
    image_folder = get_env('TEST_TREE_IMAGES_PATH')
    output_path = get_env('TEST_RESULTS_PATH')
    expert_system = ExpertSystem(image_folder, with_masks=True, chunk_size=10, start_chunk=261)
    expert_system.bulk_review()
    print("Done!")