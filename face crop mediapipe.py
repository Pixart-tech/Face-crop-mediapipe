import os
import sys
import cv2
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.python._framework_bindings import image as mp_image_module
from mediapipe.python._framework_bindings import image_frame as mp_image_frame_module

# Jawline indices (left ear → chin → right ear)
JAWLINE_IDS = [212, 93, 132, 58, 172, 136, 150, 176, 149, 152,
    377, 400, 378, 379, 365, 397, 288, 323, 432]

MPImage = mp_image_module.Image
MPImageFormat = mp_image_frame_module.ImageFormat


def init_face_landmarker(model_path="face_landmarker.task"):
    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


def crop_head_bust(image_bgr, landmarker):
    """Return cropped head bust with transparent background (everything below jaw removed)."""
    h, w = image_bgr.shape[:2]
    mp_img = MPImage(image_format=MPImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        print("❌ No face detected.")
        return None

    lm = result.face_landmarks[0]
    jaw_pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in JAWLINE_IDS], dtype=np.int32)

    # Create full head mask by extending top to cover hair region
    mask = np.zeros((h, w), np.uint8)
    head_poly = np.vstack([
        np.array([[0, 0], [w, 0], [w, jaw_pts[-1][1]]]),
        jaw_pts[::-1],
        np.array([[0, jaw_pts[0][1]]])
    ])
    cv2.fillPoly(mask, [head_poly.astype(np.int32)], 255)

    # Apply alpha
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask

    # Compute bounding box (above jaw only)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

    # Crop tightly around the head (no neck)
    cropped = bgra[y0:y1, x0:x1]
    return cropped


def process_folder(ip_dir, op_dir):
    os.makedirs(op_dir, exist_ok=True)
    landmarker = init_face_landmarker()

    for fname in os.listdir(ip_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        inp = os.path.join(ip_dir, fname)
        print("Processing:", inp)
        img = cv2.imread(inp)
        if img is None:
            print("⚠️ Could not read:", fname)
            continue

        out = crop_head_bust(img, landmarker)
        if out is not None:
            out_path = os.path.join(op_dir, os.path.splitext(fname)[0] + ".png")
            cv2.imwrite(out_path, out)
            print("✅ Saved:", out_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: py crop_head_bust_clean.py <input_dir> <output_dir>")
        sys.exit(1)
    process_folder(sys.argv[1], sys.argv[2])
