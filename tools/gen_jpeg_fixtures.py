#!/usr/bin/env python3
"""Generate small test JPEG fixtures for the rs-face decoder."""
import os
import sys
import cv2
import numpy as np

OUT_DIR = os.environ.get("OUT_DIR", "/tmp/jpeg_fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

# 8x8 grayscale constant mid-gray
img_8x8_gray = np.full((8, 8), 128, dtype=np.uint8)
result, buf = cv2.imencode('.jpg', img_8x8_gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
with open(os.path.join(OUT_DIR, "8x8_gray.jpg"), "wb") as f:
    f.write(buf.tobytes())
print(f"8x8 gray: {len(buf)} bytes")

# 16x16 grayscale with content
np.random.seed(42)
img_16x16_gray = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
result, buf = cv2.imencode('.jpg', img_16x16_gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
with open(os.path.join(OUT_DIR, "16x16_gray.jpg"), "wb") as f:
    f.write(buf.tobytes())
print(f"16x16 gray: {len(buf)} bytes")

# 16x16 RGB
img_16x16_rgb = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
result, buf = cv2.imencode('.jpg', img_16x16_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
with open(os.path.join(OUT_DIR, "16x16_rgb.jpg"), "wb") as f:
    f.write(buf.tobytes())
print(f"16x16 RGB: {len(buf)} bytes")

# 32x32 YCbCr 4:2:0 (default for cv2 RGB)
img_32x32_rgb = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
result, buf = cv2.imencode('.jpg', img_32x32_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
with open(os.path.join(OUT_DIR, "32x32_rgb.jpg"), "wb") as f:
    f.write(buf.tobytes())
print(f"32x32 RGB: {len(buf)} bytes")

# 24x24 grayscale with a face-like dark center
img_24x24 = np.full((24, 24), 200, dtype=np.uint8)
img_24x24[6:18, 8:16] = 100
result, buf = cv2.imencode('.jpg', img_24x24, [cv2.IMWRITE_JPEG_QUALITY, 90])
with open(os.path.join(OUT_DIR, "24x24_face_like.jpg"), "wb") as f:
    f.write(buf.tobytes())
print(f"24x24 face-like: {len(buf)} bytes")

print("Fixtures written to:", OUT_DIR)
