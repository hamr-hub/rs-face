# Algorithms

A reference for the algorithms implemented in `rs-face`. We follow OpenCV 4.x
semantics where applicable so that the loaded `haarcascade_frontalface_default.xml`
cascades behave identically.

## 1. Integral image (summed-area table)

For an input grayscale image `I` of size `W × H`:

```
II[x, y] = Σ I[i, j]   for 0 ≤ i < x, 0 ≤ j < y
```

with `II[0, *] = II[*, 0] = 0`. The sum over any rectangle
`[x1, x2) × [y1, y2)` is then a single O(1) lookup:

```
sum = II[x2, y2] - II[x1, y2] - II[x2, y1] + II[x1, y1]
```

We use `u32` accumulation (max ~5.3e8 for 1920×1080×255, fits).

**Two variants:**

- `IntegralImage` — stores `u32` sums.
- `SquaredIntegralImage` — stores `u64` sums of `pixel_value²`. Used for
  variance computation.

### 1a. Rotated (45°) integral image

For the tilted Haar features used in some OpenCV cascades, we also need a
"rotated" integral that sums over a 45° half-plane. We use Lienhart &
Maydt's two-pass construction:

```
Pass 1: build regular SAT, store in `sat` (a separate buffer)
Pass 2: R(x, y) = R(x-1, y-1)
              + SAT(x, y) - SAT(x-1, y)
              - SAT(x, y-1) + SAT(x-1, y-1)
        (write R into a separate `data` buffer)
```

The two-buffer split is required because pass 2 reads SAT values that
would be corrupted by an in-place transformation.

## 2. Variance pre-filter

For a window of `N` pixels, the variance is:

```
Var = E[X²] - E[X]² = (sum_sq / N) - (sum / N)²
```

A face is rejected at the variance stage iff:

```
sum_sq * N - sum² < variance_threshold * N²
```

This is the standard Viola-Jones first-stage rejection. We compute it in
O(1) using the integral images. For real images this rejects >95% of
candidate windows before the (much more expensive) cascade runs.

## 3. Multi-scale image pyramid

The cascade is trained for a single window size (24×24 for OpenCV's
frontalface). To detect faces at other sizes, we build a pyramid by
repeatedly downscaling:

```
scale[0] = 1.0
scale[i+1] = scale[i] / config.scale_factor   # default 1.2
```

For each level, we run the sliding-window detector. Per-level detections
are mapped back to the original image space via `inv_scale`.

**Resize method:** for downscaling we use area averaging (matches OpenCV's
`INTER_AREA`); for upscaling (rare) we use bilinear.

## 4. Haar-like features

Five canonical feature families (matching OpenCV):

| Family              | Layout                              | Use case          |
|---------------------|-------------------------------------|-------------------|
| VerticalEdge        | two equal horizontal rects          | vertical gradient |
| HorizontalEdge      | two equal vertical rects            | horizontal gradient |
| DiagonalEdge        | two equal 45°-tilted rects          | diagonal gradient |
| VerticalCenter      | center flanked by two side rects    | bright centre     |
| HorizontalCenter    | center flanked by two top/bottom    | bright band       |
| CustomRects         | arbitrary OpenCV-style layout       | trained cascades  |

A feature response is the weighted sum of its sub-rectangle sums:

```
response = Σ weight_i * rect_sum_i
```

Per OpenCV 4.x, we do **not** divide by a per-feature `normfactor` at eval
time (the pre-4.x convention did; many third-party ports still do — we
match the modern reference).

## 5. AdaBoost cascade

The cascade is an ordered list of `Stage`s. Each stage has a `stage_threshold`
and a list of `WeakFeature`s. A window passes a stage iff the sum of weak
feature values is ≥ the stage threshold. The window is rejected as soon as
any stage rejects it.

```
total = 0
for stage in cascade.stages:
    stage_sum = 0
    for weak in stage.weak_features:
        response = cache.get_or_eval(weak.feature_index)
        value = response * variance_norm_factor
        stage_sum += (value < weak.threshold) ? weak.left_val : weak.right_val
    if stage_sum < stage.stage_threshold: return None
    total += stage_sum
return Some(total)
```

`OpenCV sign convention`: for each weak classifier, the feature response
`value` is compared to `threshold`. If `value < threshold`, use `left_val`
(face class). Otherwise use `right_val` (non-face class). The `sign` field
on `WeakFeature` is a historical artefact from earlier OpenCV versions
and is not consulted by this implementation.

### 5a. Variance normalisation

OpenCV computes a "variance normalisation factor" once per window over the
inner rectangle `(1, 1, win_w-2, win_h-2)`:

```
var_norm = 1 / sqrt(area * sum_sq_in - sum_in²)
```

We then multiply each raw feature response by `var_norm` before thresholding.
This makes the cascade scale-invariant to illumination.

## 6. Non-maximum suppression (spatial-bucket optimisation)

The naive greedy NMS is O(n²). For large images with thousands of candidate
detections, this dominates end-to-end runtime. We use a spatial-bucket
short-circuit:

1. Sort candidates by score (descending).
2. Compute cell size = median of `max(w, h)` over all candidates.
3. Bucket each candidate into a `(bx, by)` cell.
4. For each kept box, only compare to candidates in the 3×3 bucket
   neighbourhood — boxes in non-overlapping buckets cannot have IoU > 0.

This brings typical NMS from O(n²) to O(n) amortised, with no quality
regression (the kept set is identical to naive greedy NMS).

## 7. Multi-threaded pipeline

`pipeline.rs` runs:

- **Source thread** (caller thread): reads frames, dispatches round-robin.
- **N worker threads**: each owns a `Detector` instance + `EvalCache`. Receives
  frames via bounded mpsc channels (depth = `queue_depth`).
- **Sink thread**: receives out-of-order results, reorders by `seq`, writes
  annotated PNG and `manifest.json` entry.

Backpressure: if all worker channels are full, the dispatcher blocks on
`send()`. This naturally throttles the source so we don't buffer entire
videos in memory.
