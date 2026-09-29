backend: ort
fixture: lena.ppm 512x512, 1 in frame

-- latency (ms) --
  detect (SCRFD+postprocess): n=30 mean=1605.12 p50=1632.50 p95=1959.22 min=1331.32 max=2098.47
  embed (ArcFace): n=30 mean=748.80 p50=703.55 p95=1103.69 min=603.45 max=1234.36

-- cosine similarity --
  same identity: n=20 mean=1.00 p50=1.00 p95=1.00 min=1.00 max=1.00
  different identity: n=20 mean=0.07 p50=0.07 p95=0.07 min=0.07 max=0.07
  margin (mean same - mean diff): 0.9341
  recommended threshold (midpoint worst-same/best-diff): 0.5329
  default MatchConfig threshold: 0.36 (differs from measured midpoint by > 0.10 -- consider recalibrating)
