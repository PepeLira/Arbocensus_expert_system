import os
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

from tree_reviewer.config import get_env

class TreeClassifierResnet:
    def __init__(self, image_size=540):
        self.weights_path = get_env('SPECIES_CLASSIFIER')
        self.model = torch.load(self.weights_path)
        self.clases = get_env('TREE_CLASES').split(',')
        self.im = image_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.init_model()

    def get_species(self, image):
        image = self.transform_image(image)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            predicted_class = self.clases[predicted.item()]
            confidence_score = round(probabilities[0, predicted.item()].item(), 4)
            return predicted_class, confidence_score
        
    def transform_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((self.im, self.im)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4522, 0.4760, 0.4560], std=[0.2187, 0.2252, 0.2747])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def init_model(self):
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.clases))
        self.model.load_state_dict(torch.load(self.weights_path))
        self.model = self.model.to(self.device)
        self.model.eval()
    
if __name__ == '__main__':
    classifier = TreeClassifierResnet()
    test_images = get_env('TEST_TREE_IMAGES_PATH')
    images = os.listdir(test_images)[:10]
    for image_name in images:
        image = Image.open(os.path.join(test_images, image_name))
        print(classifier.classify(image))