import os
from PIL import Image
import numpy as np
from models.tree_image import TreeImage

class ImageLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def path_to_images(self):
        files = os.listdir(self.folder_path)
        paths = []
        for file in files:
            if file.endswith(".jpg"):
                paths.append((os.path.join(self.folder_path, file), file))
        return paths

    def load_image(self, image_file=None):
        count = 1
        image_paths = self.path_to_images()
        if image_file is None:
            for image_path, file in image_paths:
                try:
                    yield self.define_image(image_path, file, count)
                    count += 1
                except FileNotFoundError:
                    print("File not found")
                except IOError:
                    print("Not a valid image file")
                continue
        else:
            try:
                image_path = os.path.join(self.folder_path, image_file)
                yield self.define_image(
                    image_path, 
                    image_file, 
                    count)
            except FileNotFoundError:
                print("File not found")
            except IOError:
                print("Not a valid image file")
        return

    def define_image(self, image_path, file_name, count):
        self.pil_image = Image.open(image_path).convert("RGB")
        self.image_array = np.array(self.pil_image)
        
        return TreeImage(self.image_array, file_name, count, self.pil_image)

            
            


if __name__ == "__main__":
    folder_path = 'C:/Users/jflir/Documents/Arbocensus/ArbocensusData/20230829_24'
    image_loader = ImageLoader(folder_path)
    
    image = next(image_loader.load_image())
    
    print(image.array)
    print(image.pil_image)
    print(image[:])
    