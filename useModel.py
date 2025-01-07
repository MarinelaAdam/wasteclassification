import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import deque
from typing import Tuple
import time
import os

CONFIDENCE_THRESHOLD = 0.6
PERSON_CLASS_ID = 0
MODEL_SAVE_PATH = "Models/mobilnetv3_93_16/best_model.pth"
CONFIDENCE_DISPLAY_THRESHOLD = 0.85

class WasteClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obj_detection_model = YOLO("yolo11x.pt")
        self.classification_model = self._load_classification_model()
        self.transform = self._get_transforms()
        self.prev_frame_time = 0

    def _load_classification_model(self) -> nn.Module:
        model = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V2')
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _classify_roi(self, roi: np.ndarray) -> Tuple[str, float]:
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.classification_model(tensor)
            prob = torch.sigmoid(output).item()
        label = "Organic" if prob < 0.5 else "Recyclable"
        return label, prob

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        self.prev_frame_time = new_frame_time

        results = self.obj_detection_model(frame)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf < CONFIDENCE_THRESHOLD or class_id == PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = box.astype(int)
            roi = frame[y1:y2, x1:x2]

            label, confidence = self._classify_roi(roi)
            color = (0, 255, 0) if label == "Recyclable" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            display_label = label
            if confidence > CONFIDENCE_DISPLAY_THRESHOLD or confidence < (1 - CONFIDENCE_DISPLAY_THRESHOLD):
                display_label += " (High Confidence)"

            cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def process_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to open.")
        return self.process_frame(image)

def main():
    classifier = WasteClassifier()

    # For single image
    image_path = r"Live_samples\img2.jpg"
    processed_image = classifier.process_image(image_path)

    # image = cv2.resize(processed_image, (512, 512))
    image = processed_image
    output_dir = "trackings"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output video path
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_video_path = os.path.join(output_dir, f"{name}_processed{ext}")

    cv2.imwrite(output_video_path, image)
    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # For video camera
    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("Error: Could not open video capture device.")
    #     return

    # print("Press 'q' to exit.")
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Error: Could not read frame.")
    #         break

    #     processed_frame = classifier.process_frame(frame)
    #     cv2.imshow("Waste Classification", processed_frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # # For video file
    # video_path = r"Live_samples\sample3.mp4"
    # cap = cv2.VideoCapture(video_path)

    # if not cap.isOpened():
    #     print("Error: Could not open video file.")
    #     return

    # # Create output directory if it does not exist
    # output_dir = "trackings"
    # os.makedirs(output_dir, exist_ok=True)

    # # Prepare output video path
    # base_name = os.path.basename(video_path)
    # name, ext = os.path.splitext(base_name)
    # output_video_path = os.path.join(output_dir, f"{name}_processed{ext}")

    # # Get video properties
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # # Define VideoWriter
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # print("Press 'q' to exit.")
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("End of video or error reading frame.")
    #         break

    #     processed_frame = classifier.process_frame(frame)
    #     out.write(processed_frame)

    #     cv2.imshow("Waste Classification", processed_frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
