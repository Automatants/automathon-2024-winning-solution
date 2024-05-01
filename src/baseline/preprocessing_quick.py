import numpy as np
import torch
import tqdm
from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FacenetDetector:
    def __init__(self, device=device) -> None:
        super().__init__()
        self.detector = MTCNN(margin=0,thresholds=[0.85, 0.95, 0.95], device=device)

    def _box_face(self, frames):
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        return [b.tolist() if b is not None else None for b in batch_boxes][0]

    def _detect(self, frames, box, margin=32):
        faces = []
        for i, b in enumerate(box):
            if b is not None:
                x1, y1, x2, y2 = b
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frames[i].shape[1], x2 + margin)
                y2 = min(frames[i].shape[0], y2 + margin)
                face = frames[i][y1:y2, x1:x2]
                #resize the face to 64x64
                face = cv2.resize(face, (64, 64))
                faces.append(face)
        return faces


class Preprocessor:
    """
    Select N frames from a given video file and transform them to tensor
    Apply then the MTCNN model
    """
    def __init__(self):
        self.margin = 32
        self.image_size = 64
        self.face_detector = FacenetDetector(device)

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

        bbox = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # change the frame size with factor 0.25
            frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

            if bbox is None:
                bbox = self.face_detector._box_face([frame])

            if bbox is None:
                continue
            face = self.face_detector._detect([frame], bbox)[0]
            frames_faces.append(face)
            count += 1

        faces = torch.stack(frames_faces, dim=0)  # tensor (time, channels, h, w
        faces = torch.permute(faces, (1, 0, 2, 3))  #tensor (channels, time, h, w)

        if save_path is not None:
            torch.save(faces, save_path)

        return faces


def main():
    data_path = "../../data/raw"
    destination_path = "../../data/processed2"

    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)

    preprocessor = Preprocessor()

    video_files = [f for f in os.listdir(data_path) if f.endswith(".mp4")]

    for file in tqdm.tqdm(video_files):
        print(file, flush=True)
        video_path = os.path.join(data_path, file)
        save_path = os.path.join(destination_path, file.replace(".mp4", ".pt"))

        if os.path.exists(save_path):
            continue

        faces = preprocessor(video_path, save_path)
        break


if __name__ == "__main__":
    print("Preprocessing the data")
    main()
    print("Preprocessing done")






