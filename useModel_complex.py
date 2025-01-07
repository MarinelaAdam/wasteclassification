import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List
from collections import deque
import time

CONFIDENCE_THRESHOLD = 0.7
MINIMUM_PERIMETER = 900
PERSON_CLASS_ID = 0
MODEL_SAVE_PATH = "Models/mobilnetv3_93_16/best_model.pth"
ROTATION_ANGLES = [-15, 0, 15]
PREDICTION_QUEUE_SIZE = 5
CONFIDENCE_DISPLAY_THRESHOLD = 0.85

class WasteClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obj_detection_model = YOLO("yolo11x.pt")
        self.classification_model = self._load_classification_model()
        self.transform = self._get_transforms()
        self.prediction_history = deque(maxlen=PREDICTION_QUEUE_SIZE)
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

    def _get_largest_valid_detection(self, boxes: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        max_perimeter = 0
        largest_box = None

        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if conf < CONFIDENCE_THRESHOLD or class_id == PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = box.astype(int)
            perimeter = 2 * (x2 - x1 + y2 - y1)

            if perimeter > max_perimeter and perimeter > MINIMUM_PERIMETER:
                max_perimeter = perimeter
                largest_box = (x1, y1, x2, y2)

        return largest_box

    # Rotate image around its center by given angle
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def _get_multi_angle_predictions(self, roi: np.ndarray) -> List[float]:
        predictions = []
        
        roi = self._preprocess_roi(roi)
        
        for angle in ROTATION_ANGLES:
            if angle == 0:
                current_roi = roi
            else:
                current_roi = self._rotate_image(roi, angle)
            
            test_roi = current_roi
            roi_pil = Image.fromarray(test_roi)
            tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.classification_model(tensor)
                prob = torch.sigmoid(output).item()
                predictions.append(prob)
        
        return predictions

    def _classify_roi(self, roi: np.ndarray) -> Tuple[str, float]:
        predictions = self._get_multi_angle_predictions(roi)
        current_prob = sum(predictions) / len(predictions)
        
        self.prediction_history.append(current_prob)
        
        # Get smoothed prediction
        avg_prob = sum(self.prediction_history) / len(self.prediction_history)
        label = "Organic" if avg_prob < 0.5 else "Recyclable"
        
        return label, avg_prob

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        self.prev_frame_time = new_frame_time
        
        results = self.obj_detection_model(frame)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        
        largest_box = self._get_largest_valid_detection(boxes, confidences, class_ids)
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if largest_box is None:
            return frame
        
        # Extract and expand ROI
        x1, y1, x2, y2 = largest_box
        width = x2 - x1
        height = y2 - y1
        
        expand_factor = 0.05
        x1 = max(0, int(x1 - width * expand_factor))
        y1 = max(0, int(y1 - height * expand_factor))
        x2 = min(frame.shape[1], int(x2 + width * expand_factor))
        y2 = min(frame.shape[0], int(y2 + height * expand_factor))
        
        roi = frame[y1:y2, x1:x2]
        label, confidence = self._classify_roi(roi)
        
        color = (0, 255, 0) if label == "Recyclable" else (0, 0, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        if confidence > CONFIDENCE_DISPLAY_THRESHOLD or confidence < (1 - CONFIDENCE_DISPLAY_THRESHOLD):
            label = f"{label} (High Confidence)"
            
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def process_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to open.")
        return self.process_frame(image)

def main():
    classifier = WasteClassifier()
    
    # For single image
    image_path = r"Live_samples\img1.jpg"
    processed_image = classifier.process_image(image_path)

    image = cv2.resize(processed_image, (512, 512))
    
    cv2.imshow("Processed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # For video
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

if __name__ == "__main__":
    main()