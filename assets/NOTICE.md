# Bundled assets — provenance & licenses

## `haarcascade_frontalface_default.rfcf`

- **Source:** OpenCV — `data/haarcascades/haarcascade_frontalface_default.xml`
  (release tag `4.10.0`).
- **Download:**
  `https://raw.githubusercontent.com/opencv/opencv/4.10.0/data/haarcascades/haarcascade_frontalface_default.xml`
- **Source SHA-256 (pinned):**
  `0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0`
  (930 127 bytes)
- **License:** Apache License 2.0 (OpenCV 4.x; © Itseez / Intel).
  The upstream XML is kept alongside the converted file for audit and
  reproducibility.
- **Conversion:** `python3 tools/convert_opencv_xml.py \
  assets/haarcascade_frontalface_default.xml \
  assets/haarcascade_frontalface_default.rfcf`
  → 25 stages, 2 913 features, no tilted rects, 24×24 window (121 200 bytes).
  The `.rfcf` format is a pure representation change (f32 weights, little
  endian); no retraining or coefficient modification is performed.

The cascade is embedded into the binary via `include_bytes!` and returned by
`rsface::haar::bundled::bundled_frontalface_cascade()`.

OpenCV's full Apache-2.0 license text: https://www.apache.org/licenses/LICENSE-2.0

## `demo_face_256.pgm`

- **Source:** area-downscaled (512×512 PPM → 256×256 P5 PGM, BT.601 luma)
  from the standard Lena test image vendored in `tests/fixtures/lena.ppm` —
  the ubiquitous 1972 image-processing test photograph, the same portrait
  OpenCV ships as `samples/data/lena.jpg`.
- **Use:** embedded into the **`rs-face` CLI binary only** (via
  `include_bytes!` in `src/main.rs`) to power the zero-argument
  `rs-face demo` install check; library embedders do not pull it in.
- **License note:** the photograph's historical copyright sits with Playboy;
  it has been used as a de-facto public-domain test fixture across the CV
  community for decades. It is shipped here solely as a functional test
  fixture. A CC0/Apache-2.0 portrait replacement is tracked so the published
  binary carries no ambiguity.
