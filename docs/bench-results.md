backend: ort
fixture: lena.ppm 512x512, 1 in frame

-- latency (ms) --
  detect (SCRFD+postprocess): n=30 mean=293.01 p50=291.08 p95=303.71 min=286.61 max=332.29
  embed (ArcFace): n=30 mean=134.61 p50=134.33 p95=141.05 min=130.86 max=145.83

-- cosine similarity --
  same identity: n=20 mean=1.00 p50=1.00 p95=1.00 min=1.00 max=1.00
  different identity: n=20 mean=0.07 p50=0.07 p95=0.07 min=0.07 max=0.07
  margin (mean same - mean diff): 0.9315
  recommended threshold (midpoint worst-same/best-diff): 0.5342
  default MatchConfig threshold: 0.36 (differs from measured midpoint by > 0.10 -- consider recalibrating)
