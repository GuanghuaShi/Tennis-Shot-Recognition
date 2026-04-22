# paste your MoveNet.py code here
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# =========================
# Global settings
# =========================
WIDTH = HEIGHT = 256

cyan = (255, 255, 0)
magenta = (255, 0, 255)
yellow = (0, 255, 255)

EDGE_COLORS = {
    (0, 1): magenta, (0, 2): cyan,
    (1, 3): magenta, (2, 4): cyan,
    (0, 5): magenta, (0, 6): cyan,
    (5, 7): magenta, (7, 9): magenta,
    (6, 8): cyan, (8, 10): cyan,
    (5, 6): yellow,
    (5, 11): magenta, (6, 12): cyan,
    (11, 12): yellow,
    (11, 13): magenta, (13, 15): magenta,
    (12, 14): cyan, (14, 16): cyan
}

# =========================
# Load MoveNet ONCE
# =========================
print("Loading MoveNet model...")
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]
print("MoveNet loaded.")

# =========================
# Drawing functions
# =========================
def draw_keypoints(frame, keypoints, threshold=0.11):
    coords = np.squeeze(np.multiply(keypoints, [HEIGHT, WIDTH, 1]))

    for y, x, conf in coords:
        if conf > threshold:
            cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 255), -1)

    return coords


def draw_edges(coords, frame, threshold=0.11):
    for (p1, p2), color in EDGE_COLORS.items():
        y1, x1, c1 = coords[p1]
        y2, x2, c2 = coords[p2]

        if c1 > threshold and c2 > threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                     color, 1, cv2.LINE_AA)


def draw_skeleton(frame, keypoints):
    for instance in keypoints:
        coords = draw_keypoints(frame, instance)
        draw_edges(coords, frame)


# =========================
# Main API (IMPORTANT)
# =========================
def run_inference(height, width, frame):
    """
    Input:
        frame (BGR image)
    Output:
        skeleton RGB image (height, width, 3)
    """

    original_size = (width, height)

    image = cv2.resize(frame, (WIDTH, HEIGHT))
    input_image = tf.cast(tf.image.resize_with_pad(image, WIDTH, HEIGHT), tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)

    results = movenet(input_image)

    keypoints = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))

    draw_skeleton(image, keypoints)

    # resize back
    image = cv2.resize(image, original_size, interpolation=cv2.INTER_LANCZOS4)

    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
