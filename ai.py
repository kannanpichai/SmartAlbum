import os
from deepface import DeepFace
import numpy as np
import hdbscan
import shutil
        
class SmartAlbum:
    def __init__(self, photos_folder, album_folder):
        self.photos_folder = photos_folder
        self.album_folder = album_folder
        self.album = None

    def make_album(self):
        embeddings = []
        photos = []

        for file_name in os.listdir(self.photos_folder):
            file_path = os.path.join(self.photos_folder,file_name)
            if os.path.isfile(file_path):
                try:
                    embedding = DeepFace.represent(img_path=file_path,model_name="Facenet512",enforce_detection=False)
                    if embedding:
                        embeddings.append(embedding[0]['embedding'])
                        photos.append(file_path)

                except Exception:
                    pass

        embeddings = np.array(embeddings)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)

        print("Labels", labels)
        grouped_images = {}
        for label, image_path in zip(labels, photos):
            if label not in grouped_images:
                grouped_images[label] = []
            grouped_images[label].append(image_path)
        
        self.album = grouped_images

    def save_album(self):
        os.makedirs(self.album_folder, exist_ok=True)

        for label, images in  self.album.items():
            if label == -1:
                continue

            group_folder = os.path.join(self.album_folder, f"group_{label}")
            os.makedirs(group_folder, exist_ok=True)
            for img_path in images:
                image_name = os.path.basename(img_path)
                dst_path = os.path.join(group_folder, image_name)
                shutil.copy(img_path,dst_path)

if __name__ == "__main__":
    PHOTOS_FOLDER = "photos"
    ALBUM_FOLDER = "album"
    ai = SmartAlbum(PHOTOS_FOLDER,ALBUM_FOLDER)
    ai.make_album()
    ai.save_album()