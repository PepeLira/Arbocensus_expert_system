from tree_reviewer.expert_system import ExpertSystem
import os
import json
import requests
from tree_reviewer.config import get_env

class ArbocensusApiInterface:
    def __init__(self, images_folder=None, output_path=os.getcwd(), chunk_size=50):
        self.images_folder = self.define_images_folder(images_folder)
        self.chunk_size = chunk_size
        self.output_path = output_path
        self.api_url = get_env('API_URL')
        self.image_count_ep = get_env('IMAGE_COUNT_EP')
        self.generate_folder_ep = get_env('GENERATE_FOLDER_EP')
        self.post_results_ep = get_env('POST_RESULTS_EP')
        self.secret_key = get_env('SECRET_KEY')
        self.marks_json_path = get_env('MARKS_JSON_PATH')
        self.current_bucket = None
        self.image_marks = {}
        self.current_folder = None
        self.json_file_name = None
        self.available_images = 0 #Number of available images to process

    def bulk_review(self, with_masks=False, start_chunk=0):
        for _ in self.image_chunks():
            self.expert_system = ExpertSystem(
                self.current_folder, 
                output_path = self.output_path, 
                with_masks = with_masks, 
                chunk_size = self.chunk_size, 
                start_chunk = start_chunk)
            self.expert_system.bulk_review(json_file_name=self.json_file_name)
            self.send_metrics()

    def image_chunks(self):
        self.get_available_images_count()
        if self.available_images == 0:
            print("No images available to process.")
            return
        # self.get_generate_folder(self.available_images)
        chunks = []
        if self.available_images < self.chunk_size:
            chunks.append(self.available_images)
        else:
            chunks = [self.chunk_size] * (self.available_images // self.chunk_size)
            if self.available_images % self.chunk_size != 0:
                chunks.append(self.available_images % self.chunk_size)
        for n in chunks:
            self.get_generate_folder(n)
            self.download_images()
            yield 

    def send_metrics(self):
        json_data = self.read_json_metrics()
        self.post_tree_metrics(json_data)

    def define_images_folder(self, images_folder):
        if images_folder is not None:
            return images_folder
        else:
            if not os.path.exists('images'):
                os.makedirs('images')
            return './images/'
    
    def get_available_images_count(self):
        image_count_url = self.api_url + self.image_count_ep
        response = requests.get(image_count_url)
        if response.status_code == 200:
            self.available_images = response.json()
            print(f"Available images: {self.available_images}")
        else:
            raise Exception(f"Error: {response.status_code}. Failed to fetch data.")

    def get_generate_folder(self, num_images):
        generate_folder_url = self.api_url + self.generate_folder_ep + f"?num_of_samples={num_images}"
        response = requests.get(
            generate_folder_url, 
            headers={'Authorization': f"Api-Key {self.secret_key}"}
        )
        if response.status_code == 200:
            response = response.json()
            self.current_bucket = "gs://"+response['bucket']
            self.image_marks.update(response['image_marks'])
            bucket_name = self.current_bucket.split('/')[-1]
            self.current_folder = self.images_folder + bucket_name
            self.json_file_name = bucket_name +'.json'
            # self.generate_marks_json()
        else:
            raise Exception(f"Error: {response.status_code}. Failed to fetch data.")
        
    def post_tree_metrics(self, json_data):
        post_results_url = self.api_url + self.post_results_ep
        response = requests.post(
            post_results_url, 
            headers={'Authorization': f"Api-Key {self.secret_key}"}, 
            json=json_data
        )
        if response.status_code == 200:
            print("Data sent successfully")
        else:
            raise Exception(f"Error: {response.status_code}. Failed to fetch data.")
        
    def generate_marks_json(self):
        # check if the file exists to create a new one
        with open(self.marks_json_path, 'w') as f:
            json.dump(self.image_marks, f, indent=4)
        
    def read_json_metrics(self):
        with open(os.path.join(self.output_path, self.json_file_name), 'r') as f:
            data = json.load(f)
            return data

    def clean_files(self):
        #Deleting the images from the folder
        os.system(f"rm -r {self.current_folder}*")
        self.join_remove_json_files()

    def join_remove_json_files(self):
        json_files = [j for j in os.listdir(self.output_path) if j.endswith('.json')]
        with open(os.path.join(self.current_folder, 'final_results.json'), 'w') as outfile:
            outfile.write('{"images": [')
            for json_file in json_files:
                with open(os.path.join(self.output_path, json_file), 'r') as infile:
                    outfile.write(infile.read())
                    outfile.write(',')
                os.remove(os.path.join(self.output_path, json_file))
            outfile.write(']}')
        
        
    def download_images(self):
        #Using the gsutil command to download the images from the bucket
        os.system(f"gsutil -m cp -r {self.current_bucket} {self.images_folder}")
        #TODO: Check if the images were downloaded successfully
        print("Images downloaded successfully")
        

if __name__ == '__main__':
    images_folder = get_env('TREE_IMAGE_DESTINY')
    arbocensus_api_interface = ArbocensusApiInterface(images_folder=images_folder, output_path=images_folder)
    arbocensus_api_interface.bulk_review(with_masks=True)
    # arbocensus_api_interface.clean_files()

    
