# `.rfcf` cascade binary format

A compact binary format for AdaBoost cascades. Used by the bundled
`haarcascade_frontalface_default.xml` after conversion via
`tools/convert_opencv_xml.py`.

## Header

```
"Magic"   4 bytes  "RFCF"
"Version" u32 LE   writer emits 3; readers accept 2 and 3
"win_w"   u32 LE   cascade window width in pixels
"win_h"   u32 LE   cascade window height in pixels
"nfeat"   u32 LE   number of features
```

## Feature records

For each of `nfeat`:

```
"kind"   u8    FeatureKind discriminant:
                0 = VerticalEdge
                1 = HorizontalEdge
                2 = DiagonalEdge
                3 = VerticalCenter
                4 = HorizontalCenter
                5 = CustomRects
"fw"     u8    feature-local width (units; 0 for CustomRects)
"fh"     u8    feature-local height (units; 0 for CustomRects)
"flags"  u8    version 3 only; bit 0 = tilted (45° rotated rects,
               evaluated against the rotated integral table).
               Other bits are reserved and must be zero.
"nrect"  u32 LE number of sub-rectangles
for each rect:
  "x"      u8    feature-local x (window-pixel x for CustomRects)
  "y"      u8    feature-local y (window-pixel y for CustomRects)
  "w"      u8    feature-local width
  "h"      u8    feature-local height
  "weight" f32 LE signed weight
```

For kinds 0–4 the `feature-local` coordinates are mapped to window
pixels at eval time: `pixel_x = x + r.x * win_w / fw`. Kind 5
(`CustomRects`, what the OpenCV converter emits) stores window-pixel
coordinates directly and uses `fw = fh = 0`.

A tilted feature (`flags & 1 == 1`, or kind 2 `DiagonalEdge`) reads
every rectangle sum from the 45° rotated integral table using
OpenCV's `CV_TILTED_OFS` corner arithmetic — see
`RotatedIntegralImage::tilted_rect_sum` in `src/integral.rs`.

## Stage records

After all features:

```
"nstages" u32 LE
for each stage:
  "stage_threshold"  f32 LE
  "n_weak"           u32 LE
  for each weak:
    "feature_index"   u32 LE   index into the feature table above
    "threshold"       f32 LE
    "sign"            i8       historical, ignored by current code
    "left_val"        f32 LE   value when response < threshold
    "right_val"       f32 LE   value when response >= threshold
```

## Examples

Convert OpenCV's classical face cascade:

```bash
python3 tools/convert_opencv_xml.py \
    /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml \
    haarcascade.rfcf
```

Use it:

```bash
./target/release/rs-face video.mp4 --out out --cascade haarcascade.rfcf
```

## Notes

- **Version 3** adds the per-feature `flags` byte so OpenCV cascades
  with `<tilted>1</tilted>` features convert faithfully. The current
  converter always writes v3.
- **Version 2** has no flags byte; every feature loads upright
  (`tilted = false`). v2 files keep loading unchanged.
- **Version 1** used a different feature layout and is rejected on
  load; regenerate old files with the current converter.
- The converter is a self-contained Python script in
  `tools/convert_opencv_xml.py` and only depends on the standard
  library.
- File size for a typical 25-stage OpenCV cascade is ~50KB.
