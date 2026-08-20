# `.rfcf` cascade binary format

A compact binary format for AdaBoost cascades. Used by the bundled
`haarcascade_frontalface_default.xml` after conversion via
`tools/convert_opencv_xml.py`.

## Header

```
"Magic"   4 bytes  "RFCF"
"Version" u32 LE   currently 2
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
"fw"     u8    feature-local width (units)
"fh"     u8    feature-local height (units)
"nrect"  u32 LE number of sub-rectangles
for each rect:
  "x"      u8    feature-local x
  "y"      u8    feature-local y
  "w"      u8    feature-local width
  "h"      u8    feature-local height
  "weight" f32 LE signed weight
```

`feature-local` coordinates are mapped to window pixels at eval time:
`pixel_x = x + r.x * win_w / fw`.

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

- **Version 1** uses a different feature layout (single packed format
  with 6-tuple rectangles). The current code reads version 2; older
  files should be regenerated via the converter.
- The converter is a self-contained Python script in
  `tools/convert_opencv_xml.py` and only depends on the standard
  library.
- File size for a typical 25-stage OpenCV cascade is ~50KB.
