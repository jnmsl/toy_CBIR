import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

IMG_SIZE = 224

_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading ResNet50...")
        _model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        print("Model ready.")
    return _model


def _l2_normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def color_histogram(hsv):
    """32-bin HSV histogram, L2-normalized."""
    h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    return _l2_normalize(np.concatenate([h, s, v]))


def lbp_descriptor(gray):
    """Uniform LBP histogram, L2-normalized."""
    lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
    hist = np.histogram(lbp.ravel(), bins=26, range=(0, 26))[0].astype("float32")
    return _l2_normalize(hist)


def cnn_features_batch(images):
    """Batch CNN feature extraction, L2-normalized."""
    model = get_model()
    batch = np.array(images, dtype="float32")
    feats = model.predict(preprocess_input(batch.copy()), verbose=0)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return feats / norms


def extract_features(img_path):
    """Extract semantic, color and texture features for a single image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    model = get_model()
    x = preprocess_input(np.expand_dims(rgb.astype("float32"), 0).copy())
    cnn_feat = model.predict(x, verbose=0).flatten()
    cnn_feat = _l2_normalize(cnn_feat)

    return {
        "semantic": cnn_feat.astype("float32"),
        "color": color_histogram(hsv).astype("float32"),
        "texture": lbp_descriptor(gray).astype("float32"),
    }
