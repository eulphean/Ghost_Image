# Segmenter benchmarks

Synthetic 640×480 frames, no camera. MediaPipe is the synchronous cost of one
frame (resize, infer, mask cleanup), which is what the worker thread has to
finish before the next frame if it is the bottleneck. Reproduce with:

```bash
python scripts/bench_segmenters.py --frames 40 --widths 256 320
```

The installation target is at least 15 FPS on a Raspberry Pi 4 or 5. The
default is MediaPipe at `processing.width: 320`. If a Pi run is under 15 FPS,
set `processing.segmenter: diff` or lower `processing.width` in
`config.local.yaml`.

## This Mac (measured)

`Darwin arm64`, Python 3.13.12, 40 frames, 25 Sep 2026.

| segmenter | width | FPS |
|-----------|------:|----:|
| diff | 256 | 170.0 |
| mediapipe | 256 | 178.7 |
| hybrid | 256 | 95.1 |
| diff | 320 | 161.9 |
| mediapipe | 320 | 258.4 |
| hybrid | 320 | 107.7 |

These numbers are well above 15 FPS, so the Mac does not force a cheaper
default. MediaPipe at width 320 was faster than 256 on this run; the selfie
model is small and the difference is inside one machine, not a reason to
change the default.

## Raspberry Pi (not measured here)

No Pi was attached when this file was written. After `./scripts/pi_setup.sh`,
run the command above on the Pi and replace this section with the printed
lines. Keep the default only if MediaPipe at width 320 is at least 15 FPS.
