"""Feature extraction for the Task 5 model, shared by the notebook and the real-time demo.

Input: one 48x48 grayscale face crop (uint8, 0-255), as in FER2013.
Output: 1,267 values = HOG (900) + LBP histograms (360) + seven geometric landmark measurements (7).

The settings match sections 1.5 (HOG), 5.4 (LBP) and 4.2 (geometric features) of
emotion_detection_group11.ipynb; section 5.6 checks that this module reproduces the notebook
features exactly.
"""
import dlib
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMAGE_SHAPE = (48, 48)
HOG_SETTINGS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                    block_norm='L2-Hys', transform_sqrt=True)
LBP_SETTINGS = {'P': 8, 'R': 1, 'method': 'uniform', 'grid_cell': 8}
# FER images are already face crops, so dlib gets a fixed box instead of a face detector.
LANDMARK_RECTANGLE = (1, 1, 47, 47)
GEOMETRIC_FEATURES = ['mouth_open', 'mouth_width', 'lip_corner_lift', 'eye_open',
                      'inner_brow_raise', 'brow_slant', 'inner_brow_gap']


def hog_features(image):
    return hog(image.astype(np.float32) / 255.0, **HOG_SETTINGS)


def lbp_features(image):
    codes = local_binary_pattern(image, LBP_SETTINGS['P'], LBP_SETTINGS['R'], LBP_SETTINGS['method'])
    bins, cell = LBP_SETTINGS['P'] + 2, LBP_SETTINGS['grid_cell']
    return np.concatenate([
        np.bincount(codes[r:r + cell, c:c + cell].astype(int).ravel(), minlength=bins) / cell ** 2
        for r in range(0, IMAGE_SHAPE[0], cell) for c in range(0, IMAGE_SHAPE[1], cell)])


def align_landmarks(points):
    """Translate to the eye midpoint, rotate the eye line horizontal, and scale by inter-ocular distance."""
    left_eye, right_eye = points[36:42].mean(0), points[42:48].mean(0)
    dx, dy = right_eye - left_eye
    iod = np.hypot(dx, dy)
    cos, sin = dx / iod, dy / iod
    rotation = np.array([[cos, sin], [-sin, cos]])
    return (points - (left_eye + right_eye) / 2) @ rotation.T / iod


def geometric_features(points):
    """The seven Task 4 measurements, in GEOMETRIC_FEATURES order (image y points down)."""
    p = align_landmarks(points)
    dist = lambda a, b: np.linalg.norm(p[a] - p[b])
    eye_width = (dist(36, 39) + dist(42, 45)) / 2
    return np.array([
        np.mean([dist(61, 67), dist(62, 66), dist(63, 65)]),
        dist(48, 54),
        (p[51, 1] + p[57, 1]) / 2 - (p[48, 1] + p[54, 1]) / 2,
        (dist(37, 41) + dist(38, 40) + dist(43, 47) + dist(44, 46)) / (4 * eye_width),
        p[[37, 38, 43, 44], 1].mean() - p[20:24, 1].mean(),
        p[[17, 26], 1].mean() - p[[21, 22], 1].mean(),
        dist(21, 22),
    ])


class FeatureExtractor:
    """Turns 48x48 uint8 face crops into the Task 5 feature vector."""

    def __init__(self, predictor_path):
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.rectangle = dlib.rectangle(*LANDMARK_RECTANGLE)

    def landmark_points(self, image):
        shape = self.predictor(np.ascontiguousarray(image, dtype=np.uint8), self.rectangle)
        return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)

    def __call__(self, image):
        if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
            raise ValueError(f'Expected a {IMAGE_SHAPE} uint8 image, got {image.shape} {image.dtype}')
        return np.concatenate([
            hog_features(image).astype(np.float32),
            lbp_features(image).astype(np.float32),
            geometric_features(self.landmark_points(image)).astype(np.float32),
        ])

    def transform(self, images):
        return np.stack([self(image) for image in images])
