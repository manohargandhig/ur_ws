import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import zmq
import pickle
import cv2
import numpy as np
import sys

np.random.seed(42)

# ---------------- MaskRCNN Setup ----------------
ROOT_DIR = "/home/roboticslab/Mask_RCNN-master/Mask_RCNN-master"
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import mrcnn.model as modellib
from mrcnn.config import Config

MODEL_DIR    = os.path.join(ROOT_DIR, "logs")
WEIGHTS_PATH = os.path.join(
    ROOT_DIR, "logs",
    "object20260305T1222",
    "mask_rcnn_object_0090.h5"
)

class CustomConfig(Config):
    NAME                     = "object"
    GPU_COUNT                = 1
    IMAGES_PER_GPU           = 1
    NUM_CLASSES              = 1 + 5
    DETECTION_MIN_CONFIDENCE = 0.55
    IMAGE_MIN_DIM            = 800
    IMAGE_MAX_DIM            = 1024
    RPN_ANCHOR_SCALES        = (8, 16, 32, 64, 128)

CLASS_NAMES = ["BG", "scissor", "stapler", "calculator", "cube", "mouse"]

CLASS_COLORS = {
    name: tuple(int(c) for c in np.random.randint(60, 220, 3))
    for name in CLASS_NAMES
}

# ---------------- Load Model ----------------
config = CustomConfig()
model  = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
print("Loading weights:", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name=True)
print("Model loaded successfully")

# ---------------- ZMQ Setup ----------------
context = zmq.Context()

sub_sock = context.socket(zmq.SUB)
sub_sock.connect("tcp://localhost:5555")
sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")
sub_sock.setsockopt(zmq.RCVHWM, 2)

pub_sock = context.socket(zmq.PUB)
pub_sock.bind("tcp://*:5556")

print("Waiting for RGBD bundles...")

# ---------------- Helper: unpack incoming message ----------------
def unpack(raw):
    obj = pickle.loads(raw)
    if isinstance(obj, dict) and "rgb" in obj:
        bgr        = obj["rgb"]
        depth      = obj["depth"]
        cam_info   = obj["cam_info"]
        depth_info = obj["depth_info"]
        stamp      = obj.get("stamp", 0.0)
        if depth is not None and depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
    else:
        bgr        = obj
        depth      = None
        cam_info   = None
        depth_info = None
        stamp      = 0.0
    return bgr, depth, cam_info, depth_info, stamp

# ---------------- Helper: depth stats inside mask ----------------
def depth_stats_in_mask(depth, mask, roi):
    if depth is None:
        return None
    y1, x1, y2, x2 = roi
    roi_depth = depth[y1:y2, x1:x2]
    mask_roi  = mask[y1:y2, x1:x2]
    valid     = roi_depth[mask_roi & (roi_depth > 0.1) & (roi_depth < 10.0)]
    if valid.size == 0:
        return None
    return {
        "mean_m":   float(valid.mean()),
        "median_m": float(np.median(valid)),
        "min_m":    float(valid.min()),
        "max_m":    float(valid.max()),
    }

# ---------------- Filter: reject robot base misdetected as cube -----------
def is_robot_base_false_detection(label, centroid, bbox, ds, img_shape):
    """
    The robot base (blue cylinder) is consistently misclassified as 'cube'.
    Reject the detection if ANY of these conditions are true:

      1. Centroid is in the bottom 35% of the image
         (robot base always appears low in the D435i view)
      2. Depth < 0.65m
         (robot base is much closer than table objects)
      3. Bounding box covers > 20% of the image area
         (robot base bbox is abnormally large)
    """
    if label != "cube":
        return False   # only filter cube misdetections

    img_h, img_w    = img_shape[:2]
    u, v            = centroid
    y1, x1, y2, x2 = bbox

    # Condition 1: vertical position
    if v > img_h * 0.65:
        #print(f"  [FILTER] cube @ v={v} rejected -- in robot base zone "
        #      f"(bottom 35%, threshold v>{int(img_h * 0.65)})")
        return True

    # Condition 2: too close
    if ds is not None and ds["median_m"] < 0.65:
        #print(f"  [FILTER] cube rejected -- depth {ds['median_m']:.3f}m < 0.65m")
        return True

    # Condition 3: bbox too large
    bbox_area  = float((y2 - y1) * (x2 - x1))
    image_area = float(img_h * img_w)
    if bbox_area / image_area > 0.20:
        #print(f"  [FILTER] cube rejected -- bbox is "
        #      f"{bbox_area / image_area * 100:.1f}% of image (> 20%)")
        return True

    return False

# ---------------- Inference Loop ----------------
while True:
    raw = sub_sock.recv()
    bgr, depth, cam_info, depth_info, stamp = unpack(raw)

    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = model.detect([rgb], verbose=0)
    r       = results[0]
    print(f"[{stamp:.3f}] Raw detections: {len(r['scores'])}")

    overlay    = bgr.copy()
    detections = []

    for i in range(len(r["scores"])):
        score = float(r["scores"][i])
        if score < 0.5:
            continue

        y1, x1, y2, x2 = r["rois"][i]
        class_id        = int(r["class_ids"][i])
        label           = CLASS_NAMES[class_id]
        mask            = r["masks"][:, :, i]
        color           = CLASS_COLORS[label]

        ys, xs   = np.where(mask)
        centroid = (int(xs.mean()), int(ys.mean())) if xs.size > 0 else (
            (x1 + x2) // 2, (y1 + y2) // 2
        )

        ds = depth_stats_in_mask(depth, mask, (y1, x1, y2, x2))

        # ---- Robot base filter ----
        if is_robot_base_false_detection(label, centroid, (y1, x1, y2, x2), ds, bgr.shape):
            continue   # skip this detection entirely

        overlay[mask] = color

        detections.append({
            "bbox":        [y1, x1, y2, x2],
            "label":       label,
            "class_id":    class_id,
            "score":       score,
            "mask":        mask.astype(np.uint8),
            "centroid_px": centroid,
            "depth_stats": ds,
        })

    # ---------------- Visualise ----------------
    vis = cv2.addWeighted(overlay, 0.5, bgr, 0.5, 0)

    for det in detections:
        y1, x1, y2, x2 = det["bbox"]
        label           = det["label"]
        score           = det["score"]
        color           = CLASS_COLORS[label]
        cx, cy          = det["centroid_px"]
        ds              = det["depth_stats"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {score:.2f}"
        if ds:
            text += f" | {ds['median_m']:.3f}m"

        cv2.putText(vis, text, (x1, max(y1 - 10, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.circle(vis, (cx, cy), 4, color, -1)

    # ---------------- Publish to pose node ----------------
    msg = {
        "stamp":      stamp,
        "frame":      vis,
        "detections": detections,
        "depth":      depth,
        "cam_info":   cam_info,
        "depth_info": depth_info,
    }
    pub_sock.send(pickle.dumps(msg, protocol=4))
    print(f"Published {len(detections)} detections")

    # ---------------- Display ----------------
    cv2.imshow("Mask R-CNN Detection", vis)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
