import os
from PIL import Image
import numpy as np
from tree_reviewer.models.tree_image import TreeImage
from datetime import datetime
from tree_reviewer.config import get_env
import json

class ImageLoader:
    def __init__(self, folder_path, chunk_size, start_chunk):
        self.folder_path = folder_path
        self.n_images = None # Number of images in the folder
        self.image_paths = self.path_to_images()
        self.chunk_size = chunk_size
        self.data_chunks = self.define_data_chunks(self.image_paths, chunk_size)
        self.current_chunk = self.data_chunks[start_chunk]
        self.current_date = datetime.now().date().__str__() 
        self.marks_json_path = get_env('MARKS_JSON_PATH')
        self.marks_data = self.define_marks()
    
    def define_marks(self):
        # check if the marks json file exists
        if os.path.exists(self.marks_json_path):
            with open(self.marks_json_path, 'r') as f:
                marks_data = json.load(f)
        else:
            marks_data = None
        return marks_data

    def path_to_images(self):
        files = os.listdir(self.folder_path)
        paths = []
        for file in files:
            if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".png"):
                paths.append((os.path.join(self.folder_path, file), file))

        self.n_images = len(paths)
        return paths
    
    def define_data_chunks(self, files, chunk_size):
        data_chunks = []
        print(f"Number of images: {len(files)}")
        for i in range(0, len(files), chunk_size):
            if i + chunk_size > len(files):
                data_chunks.append(files[i:])
            data_chunks.append(files[i:i+chunk_size])
        return data_chunks
    
    def next_chunk(self):
        if len(self.data_chunks) > 1:
            self.current_chunk = self.data_chunks.pop(0)
        else:
            print("End of program")

    def load_image(self, image_file=None):
        self.count = 1
        if image_file is None:
            for image_path, file in self.current_chunk:
                try:
                    yield self.define_image(image_path, file, self.count)
                    self.count += 1
                except FileNotFoundError:
                    print(f"File not found: {file}")
                except IOError:
                    print(f"Not a valid image file: {file}")
                continue
        else:
            try:
                image_path = os.path.join(self.folder_path, image_file)
                yield self.define_image(
                    image_path, 
                    image_file, 
                    self.count)
            except FileNotFoundError:
                print("File not found")
            except IOError:
                print("Not a valid image file")

    def define_image(self, image_path, file_name, count):
        self.pil_image = Image.open(image_path).convert("RGB")
        self.image_array = np.array(self.pil_image)
        file_id = file_name.split('.')[0]
        if file_id in list(self.marks_data.keys()):
            card_mark = self.marks_data[file_id][0]
        else:
            card_mark = None

        return TreeImage(
            self.image_array, 
            file_name, 
            count, 
            self.pil_image, 
            card_mark = card_mark
        )
    
    def current_chunk_name(self):
        return 'batch_' + self.current_date + '_'

if __name__ == "__main__":
    folder_path = 'C:/Users/jflir/Documents/Arbocensus/ArbocensusData/20230829_24'
    image_loader = ImageLoader(folder_path)
    
    image = next(image_loader.load_image())
    
    print(image.array)
    print(image.pil_image)
    print(image[:])
    