"""Train/test-independent 64px grayscale pixel features; PCA is fitted by the caller on train only."""
import numpy as np
from PIL import Image

def load_features(records, data, channel, size, classes):
    x = np.empty((len(records), size * size), dtype=np.float32)
    y = np.empty(len(records), dtype=np.int64)
    for i, r in enumerate(records):
        path = data/'images'/'clean'/channel/r['class_name']/(r['record_id']+'.png')
        with Image.open(path) as image:
            image = image.convert('L').resize((size, size), Image.Resampling.BILINEAR)
            x[i] = np.asarray(image, dtype=np.float32).reshape(-1) / 255.
        y[i] = classes.index(r['class_name'])
    return x, y
