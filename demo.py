"""Real-time emotion recognition with the Task 5 model (HOG + LBP + geometric features).

Usage (from any folder):
    python demo.py                    # default webcam
    python demo.py --camera 1         # another camera
    python demo.py --no-smoothing     # predict every frame independently
Press q to quit.

Adapted from the supplied baseline_demo_fer2013.py. Features come from src/fer_features.py,
the module used to train and test the model (section 5.6 of emotion_detection_group11.ipynb).
Adaptations for webcam use (section 6 of the notebook):
- the face box is shrunk by 10% per side, which matches FER2013's tight framing best;
- probabilities are averaged over recent frames, so the label does not flicker;
- only the largest face is classified, ignoring small false detections in the background.
"""
import argparse
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np

from src.fer_features import FeatureExtractor

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models' / 'task5' / 'hog_lbp_geometric_random_forest.pkl'
FACE_CASCADE = ROOT / 'faceAndLandmarkModels' / 'haarcascade_frontalface_default.xml'
CROP_MARGIN = -0.10      # fraction of the box added per side; negative crops tighter
SMOOTHING = 0.3          # weight of the newest frame in the running average of probabilities
MIN_FACE_SIZE = (80, 80)  # pixels; smaller detections are usually background false positives


class EmotionRecognizer:
    def __init__(self, model_path=DEFAULT_MODEL, margin=CROP_MARGIN, smoothing=SMOOTHING):
        bundle = joblib.load(model_path)
        self.model = bundle['classifier']
        # One face at a time: parallel prediction only adds overhead per call.
        if hasattr(self.model, 'n_jobs'):
            self.model.n_jobs = 1
        self.emotions = bundle['emotion_names']
        self.extractor = FeatureExtractor(ROOT / bundle['landmark_predictor'])
        self.detector = cv2.CascadeClassifier(str(FACE_CASCADE))
        if self.detector.empty():
            raise FileNotFoundError(f'Could not load the face detector: {FACE_CASCADE}')
        self.margin, self.smoothing = margin, smoothing
        self.average = None

    def probabilities(self, gray, box):
        """Emotion probabilities for one detected face box in a grayscale frame."""
        x, y, w, h = box
        dx, dy = int(w * self.margin), int(h * self.margin)
        face = gray[max(0, y - dy):min(gray.shape[0], y + h + dy),
                    max(0, x - dx):min(gray.shape[1], x + w + dx)]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        return self.model.predict_proba(self.extractor(face).reshape(1, -1))[0]

    def process(self, frame):
        """Detect the largest face and return (box, emotion, probability), or None if no face."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5,
                                               minSize=MIN_FACE_SIZE)
        if len(faces) == 0:
            self.average = None  # start fresh when the face returns
            return None
        box = max(faces, key=lambda b: b[2] * b[3])
        current = self.probabilities(gray, box)
        if self.smoothing is None or self.average is None:
            self.average = current
        else:
            self.average = self.smoothing * current + (1 - self.smoothing) * self.average
        best = int(np.argmax(self.average))
        return box, self.emotions[best], float(self.average[best])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--camera', type=int, default=0, help='webcam index (default 0)')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='model bundle (.pkl)')
    parser.add_argument('--margin', type=float, default=CROP_MARGIN,
                        help=f'face box margin per side (default {CROP_MARGIN})')
    parser.add_argument('--no-smoothing', action='store_true', help='predict every frame independently')
    args = parser.parse_args()

    print('Loading model...')
    recognizer = EmotionRecognizer(args.model, args.margin, None if args.no_smoothing else SMOOTHING)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {args.camera}')
    print('Running. Press q in the video window to quit.')

    fps = None
    while True:
        start = perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        result = recognizer.process(frame)
        if result is not None:
            (x, y, w, h), emotion, probability = result
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'{emotion} {probability:.0%}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        frame_fps = 1 / max(perf_counter() - start, 1e-6)
        fps = frame_fps if fps is None else 0.9 * fps + 0.1 * frame_fps
        cv2.putText(frame, f'{fps:.0f} FPS', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
