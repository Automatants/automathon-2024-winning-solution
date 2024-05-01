import numpy as np
import torch.backends.mps
import tqdm
from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
print(f"running on device: {device}")

class Preprocessor:
    """
    Select N frames from a given video file and transform them to tensor
    Apply then the MTCNN model
    """
    def __init__(self):
        self.margin = 32
        self.image_size = 64
        self.face_detector = MTCNN(select_largest=False,
                                   margin=self.margin,
                                   image_size=self.image_size,
                                   thresholds=[0.3, 0.35, 0.35],
                                   )
        self.batch_size = 64

    def __call__(self, video_path, save_path=None):
        """
        Parameters:
        ----------
            video_path (str): path to the video file
            save_path (str): path to save the tensor

        Returns:
        -------
            faces (torch.Tensor): tensor of shape (channels, n_frames, h, w)
        """
        cap = cv2.VideoCapture(video_path)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        count = 0

        frames = []
        frames_faces = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # change the frame size with factor 0.25
            frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            frames.append(frame)
            count += 1
            if count % self.batch_size == 0 or count == n_frames:
                faces = self.face_detector(frames)  # tensor (time, channels, h, w)
                frames_faces.extend(faces)
                frames = []

        faces = torch.stack(frames_faces, dim=0)  # tensor (time, channels, h, w
        faces = torch.permute(faces, (1, 0, 2, 3))  #tensor (channels, time, h, w)

        if save_path is not None:
            torch.save(faces, save_path)

        return faces

if __name__ == "__main__":
    data_path = "../../data/raw"
    destination_path = "../../data/processed"

    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)

    preprocessor = Preprocessor()

    for file in tqdm.tqdm([f for f in os.listdir(data_path) if f.endswith(".mp4")]):
        video_path = os.path.join(data_path, file)
        save_path = os.path.join(destination_path, file.replace(".mp4", ".pt"))

        try:
            faces = preprocessor(video_path, save_path)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue








