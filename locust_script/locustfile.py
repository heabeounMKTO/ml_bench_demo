from locust import HttpUser, task, between, TaskSet
import time 
import os 
import random
from common_util import get_all_image_from_dir

all_images = get_all_image_from_dir("/home/hbdesk/Pictures")

class FaceExtract(TaskSet):
    @task 
    def get_face(self):
        random_image = all_images[random.randint(0, len(all_images))]  
        print(random_image)
        random_image_path=os.path.join("/home/hbdesk/Pictures", random_image)
        with open(random_image_path, 'rb') as img_file:
            files = {'input': ('image.jpg', img_file, 'image/jpeg')}
            self.client.post("/get_face", files=files, verify=False)


class APIEndpoint(HttpUser):
    tasks = [FaceExtract]
    wait_time=between(1,3)
    host="http://0.0.0.0:9995"
