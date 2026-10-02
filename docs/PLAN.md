# Ghost Image — Development Plan

This plan breaks the project into small, independently testable steps. Each step
has a goal, the work involved, and a "done when" check. Steps are ordered so
that something runnable exists after every step, and so the Mac → Raspberry Pi
handoff is exercised early rather than at the end.

---

## 0. Understanding of the project

- A USB camera on a Raspberry Pi looks at a fixed scene in an exhibition space.
- The operator presses a key to **hold** the current frame. That frame becomes
  the permanent background reference.
- Visitors walking into view are **segmented** from the live feed (body, arms,
  hands, fingers) and everything except them is discarded.
- The segmented person is **composited** onto the held frame as a translucent
  ghost with a glowing outline. The held scene remains visible through them.
- Deliverable: a single Python/OpenCV app that runs unchanged on macOS (dev)
  and Raspberry Pi (production), with USB camera auto‑detection and kiosk
  autostart on the Pi.

Pipeline per frame once a frame is held:

```
camera frame ──► person mask (segmentation) ──► feather/clean mask
                                                       │
held frame ─────────────────────────────────────────►  composite:
                                                       held*(1-α·mask) + ghost(frame)*(α·mask)
                                                       + glow(contour(mask))
                                                       ──► display
```

---

## Phase 1 — Project scaffold and environment

### Step 1.1 — Repository skeleton
- Create the package layout described in `AGENTS.md`
  (`ghost_image/`, `docs/`, `scripts/`, `tests/`, `models/`).
- Add `requirements.txt` (pinned): `opencv-python`, `numpy`, `mediapipe`,
  `pyyaml`, `pytest`, `black`, `ruff`.
- Add `.gitignore` (venv, `__pycache__`, `models/*.tflite`, captures).
- Add `README.md` with Mac setup instructions (`python3.11 -m venv .venv`,
  `pip install -r requirements.txt`, `python -m ghost_image`).
- **Done when:** `python -m ghost_image` prints a version string and exits;
  `pytest` runs (zero tests is fine).

### Step 1.2 — Configuration module
- `config.py` loads `config.yaml`, applies defaults, validates types.
- Fields: camera (preferred name/index, width, height, fps), processing
  resolution, ghost alpha, glow colour/thickness/blur, model path, fullscreen,
  key bindings, debug flag.
- **Done when:** unit tests cover defaults, overrides and invalid values.

### Step 1.3 — Python version note
- **Resolved:** MediaPipe 1.0.1 ships `py3-none` wheels (any Python 3) for
  both macOS arm64 and Linux aarch64, and was verified to install and import
  alongside OpenCV 5.0.0 on Python 3.13 and 3.14 on the Mac. No separate
  3.11 install is needed.
- The project requires Python ≥ 3.11 (`pyproject.toml`). Development uses the
  Homebrew Python 3.13 already on the Mac; Raspberry Pi OS Trixie ships 3.13
  and Bookworm ships 3.11, so both are supported by `scripts/pi_setup.sh`.

**Phase 1 status: complete.** `.venv` created with Python 3.13, pinned
`requirements.txt`, `config.py` + `config.yaml` with 41 passing tests,
`python -m ghost_image` reports version/environment, `ruff` and `black` clean.

---

## Phase 2 — Camera: detect USB device and show the live feed (Requirement 1)

### Step 2.1 — Cross‑platform camera discovery (`camera.py`)
- `list_cameras()` returns candidate devices with a stable description.
  - **Linux (Pi):** scan `/dev/video*`; for each, read
    `/sys/class/video4linux/videoN/device` and keep only devices whose real
    path is under a USB bus and that advertise video capture (`v4l2` caps;
    skip metadata‑only nodes). Prefer the lowest‑numbered capture node of a
    USB device. Open with `cv2.CAP_V4L2`, request MJPG for higher FPS.
  - **macOS:** probe indices 0..N with `cv2.CAP_AVFOUNDATION` and keep those
    that return a frame. Allow a config override to pick a specific index
    (e.g. skip the built‑in FaceTime camera).
- `open_camera(config)` picks the best match, sets resolution/FPS, and verifies
  a frame is actually returned before declaring success.
- Reconnect logic: if `read()` fails N times in a row, close and re‑run
  discovery (USB cameras get unplugged in exhibitions).
- **Done when:** on the Mac, the app opens the external USB webcam when one is
  plugged in and falls back to the built‑in camera otherwise; a `--list-cameras`
  CLI flag prints what it found.

### Step 2.2 — Live view window (`ui.py`, `app.py`)
- Main loop: grab frame → (no processing yet) → show in a named window.
- FPS counter overlay, fullscreen toggle (`F`), quit (`Q`/Esc).
- Graceful shutdown (release camera, destroy windows) on Ctrl‑C too.
- **Done when:** stable live feed at the camera's native FPS on the Mac.

### Step 2.3 — First Raspberry Pi smoke test
- Write `scripts/pi_setup.sh`: apt packages (`python3-venv`, `libatlas`,
  `libgl1`, `v4l-utils`), create venv, pip install.
- Clone on the Pi, run, confirm USB camera detection and live view.
- Record baseline FPS and CPU usage; this decides processing resolution later.
- **Done when:** live feed runs on the Pi from a fresh clone using only the
  README instructions.

---

## Phase 3 — Hold a reference frame (Requirement 2)

### Step 3.1 — Application state machine
- States: `LIVE` (pass‑through) and `HELD` (ghost mode).
- `SPACE` captures the current frame into `held_frame` and switches to `HELD`.
- `R` clears it and returns to `LIVE`.
- Optionally average several consecutive frames when holding to reduce sensor
  noise in the reference.
- **Done when:** pressing SPACE freezes the displayed image; R returns to live.

### Step 3.2 — Persistence
- Save the held frame to disk (`captures/held.png`) so the installation can
  recover after a power cycle; `--restore-held` flag or config option loads it
  at start.
- Show a small on‑screen indicator (corner dot or text) of the current state.
- **Done when:** restart the app and the held frame is back without operator
  action.

---

## Phase 4 — Person segmentation (Requirement 3)

### Step 4.1 — Segmentation interface
- `segmentation.py` exposes `Segmenter.mask(frame) -> np.ndarray` returning a
  float32 mask in `[0,1]` at frame resolution (1 = person).
- Implementations are swappable via config so they can be compared on the Pi.

### Step 4.2 — MediaPipe Image Segmenter (primary)
- Use the MediaPipe Tasks `ImageSegmenter` with the selfie/person model
  (`selfie_segmenter.tflite`, and evaluate the multiclass model for better
  hand/finger coverage). Add `scripts/download_models.sh`.
- Run inference at processing resolution (e.g. 256–320 px wide), upscale the
  confidence mask to display size.
- Run in `LIVE_STREAM` or `VIDEO` mode with a worker thread so capture isn't
  blocked by inference.
- **Done when:** debug view shows a clean person mask that tracks arms and
  hands on the Mac.

### Step 4.3 — Frame‑difference fallback / refinement
- Because a held reference exists, compute `|frame - held_frame|` (in a
  lighting‑robust space such as Lab or with per‑frame gain normalisation),
  threshold, and morphologically clean it.
- Use it (a) as a fallback if MediaPipe is unavailable/too slow on the target
  Pi, and (b) optionally intersect/union with the ML mask to recover thin
  fingers or reject false positives.
- **Done when:** the fallback alone produces a usable (if rougher) mask, and
  the config can select `mediapipe`, `diff`, or `hybrid`.

### Step 4.4 — Mask post‑processing
- Temporal smoothing (EMA over 2–3 frames) to stop flicker.
- Morphological open/close, small‑blob removal, Gaussian feathering of edges.
- Expose all thresholds in config; show mask in the debug view.
- **Done when:** the mask is stable with soft edges and no flickering specks.

### Step 4.5 — Segmentation on the Pi
- Benchmark each segmenter on the Pi at several resolutions; pick defaults that
  achieve ≥ 15 FPS. Document the numbers in `docs/BENCHMARKS.md`.
- **Done when:** the chosen default runs at target FPS on the Pi.

---

## Phase 5 — Ghost compositing (Requirement 4)

### Step 5.1 — Base blend (`compositor.py`)
- `composite(held, frame, mask, params) -> image`.
- `out = held * (1 - α·mask) + ghost(frame) * (α·mask)` where `ghost()`
  applies an optional stylisation to the person pixels: desaturate, cool tint,
  slight brightness lift, mild blur.
- α (opacity) adjustable live with `[`/`]`.
- **Done when:** a person appears translucent with the held scene visible
  through them.

### Step 5.2 — Contour glow
- Extract contours from the thresholded mask (`cv2.findContours`) or use the
  mask's gradient/edge band.
- Draw the outline with configurable thickness, Gaussian‑blur it into a halo,
  and add it to the composite (screen/additive blend) in a configurable colour.
- Optionally animate glow intensity slowly (sine pulse) for a spectral feel.
- **Done when:** the silhouette has a soft luminous rim on all limbs and
  fingers.

### Step 5.3 — Visual tuning tools
- Debug view cycle (`D`): live feed → mask → held frame → composite.
- On‑screen readout of current α, glow settings, FPS, segmenter name.
- Save a snapshot of the composite with `S` for reviewing looks offline.
- **Done when:** all look parameters can be tuned without editing code.

---

## Phase 6 — Robustness for an exhibition

### Step 6.1 — Failure handling
- Camera disconnect → show "camera lost" screen, retry discovery in a loop.
- Segmenter exception → fall back to `diff` segmenter and log once.
- Frame timeouts never crash the app; the main loop is wrapped and logged.
- **Done when:** unplugging and replugging the camera recovers automatically.

### Step 6.2 — Lighting drift
- Exhibition lighting changes across the day. Add an optional slow background
  update for the *diff* segmenter (only where no person is detected), while
  keeping the displayed held frame fixed.
- **Done when:** the diff mask doesn't fill with noise after lighting changes.

### Step 6.3 — Logging and performance
- Rotating log file; per‑stage timing in debug mode.
- Optional frame skipping for segmentation (segment every 2nd frame, reuse
  mask) as a Pi performance lever.

---

## Phase 7 — Raspberry Pi deployment

### Step 7.1 — Kiosk autostart
- `scripts/install_service.sh` installs a systemd user service that starts the
  app fullscreen on boot, restarts on failure, and waits for the display.
- Disable screen blanking; hide cursor.
- **Done when:** Pi boots straight into the installation with no keyboard.

### Step 7.2 — Operator controls without keyboard (optional)
- Support a GPIO button or a small USB keypad for HOLD / RELEASE so staff can
  re‑hold the frame if the camera is bumped.

### Step 7.3 — Final documentation
- README: Mac dev setup, Pi setup, operating instructions, troubleshooting
  (camera not found, low FPS, model download).
- `docs/BENCHMARKS.md` with measured FPS per configuration.

---

## Testing strategy

- **Unit tests (no camera):** config loading, mask post‑processing on synthetic
  masks, compositor math on synthetic images (alpha correctness, glow only on
  edges), camera discovery parsing with a mocked sysfs tree.
- **Manual checks per phase:** listed as "Done when" above.
- **Pi checks:** after Phases 2, 4 and 7, run on the actual Pi from a clean
  clone.

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| MediaPipe too slow / unavailable on the Pi | `diff` and `hybrid` segmenters; frame skipping; lower processing resolution |
| Thin fingers lost by the segmentation model | Hybrid mask with frame difference; mask dilation tuned per venue |
| Camera index differs between Mac and Pi | Discovery by device type, config override, never hardcoded |
| Lighting changes break the diff mask | Slow background adaptation outside person regions |
| Camera bumped after holding | Persisted held frame + easy re‑hold control |
| Python version mismatch | Pin 3.11 on both platforms, documented in README |

## Suggested order of work

1.1 → 1.2 → 2.1 → 2.2 → 2.3 (first Pi test) → 3.1 → 3.2 → 4.1 → 4.2 → 4.4 →
5.1 → 5.2 → 5.3 → 4.3 → 4.5 (Pi benchmark) → 6.x → 7.x
