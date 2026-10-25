"""Real-time emotion recognition with the Task 5 model (HOG + LBP + geometric features).

Usage (from any folder):
    python demo.py                 # default webcam
    python demo.py --camera 1      # another camera
Press q to quit.

Adapted from the supplied baseline_demo_fer2013.py: the face detection and drawing are the
same, but features come from src/fer_features.py, the module used to train and test the model
in section 5.6 of emotion_detection_group11.ipynb.
"""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.fer_features import FeatureExtractor

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models' / 'task5' / 'hog_lbp_geometric_random_forest.pkl'
FACE_CASCADE = ROOT / 'faceAndLandmarkModels' / 'haarcascade_frontalface_default.xml'


class EmotionRecognizer:
    def __init__(self, model_path=DEFAULT_MODEL):
        bundle = joblib.load(model_path)
        self.model = bundle['classifier']
        # One face at a time: parallel prediction only adds overhead per call.
        if hasattr(self.model, 'n_jobs'):
            self.model.n_jobs = 1
        self.emotions = bundle['emotion_names']
        self.extractor = FeatureExtractor(ROOT / bundle['landmark_predictor'])

    def predict(self, face_gray):
        """Return (emotion, probability) for a grayscale face crop of any size."""
        face = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA).astype(np.uint8)
        probabilities = self.model.predict_proba(self.extractor(face).reshape(1, -1))[0]
        best = int(np.argmax(probabilities))
        return self.emotions[best], float(probabilities[best])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--camera', type=int, default=0, help='webcam index (default 0)')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='model bundle (.pkl)')
    args = parser.parse_args()

    print('Loading model...')
    recognizer = EmotionRecognizer(args.model)
    face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE))
    if face_cascade.empty():
        raise FileNotFoundError(f'Could not load the face detector: {FACE_CASCADE}')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {args.camera}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            emotion, probability = recognizer.predict(gray[y:y + h, x:x + w])
            cv2.putText(frame, f'{emotion} {probability:.0%}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
