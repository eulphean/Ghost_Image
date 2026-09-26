# Ghost Image — Agent Guide

## What this project is

Ghost Image is an interactive art installation built in Python with OpenCV. A USB
camera points at a fixed scene. The operator "holds" a reference frame of that
scene. From then on, any person who walks into the camera's view is rendered as a
translucent, edge‑lit **ghost** composited on top of the held reference frame, so
the original scene remains visible *through* their body, hands and fingers.

The application is developed on macOS and deployed to a Raspberry Pi (64‑bit
Raspberry Pi OS) with a USB webcam. All code must run on both platforms without
modification; platform differences are isolated behind small abstractions.

## Core behaviour (the four requirements)

1. **Live feed** — open the USB camera automatically (detect it, don't hardcode an
   index) and show the live feed in a window.
2. **Hold a frame** — on operator command (keyboard/GPIO), capture the current
   frame as the persistent *background reference*.
3. **Ghost people** — once a frame is held, detect people (whole body, arms,
   hands, fingers) in each new frame and remove everything that is not the
   person (background removal / person segmentation).
4. **Composite** — blend the segmented person semi‑transparently over the held
   reference frame and highlight their silhouette contour with a glowing edge so
   they read as a ghost hovering over the original scene.

## Key technical decisions

- **Language/runtime:** Python ≥ 3.11, developed on 3.13 (Homebrew on the Mac;
  Raspberry Pi OS Trixie ships 3.13, Bookworm 3.11). MediaPipe 1.0.1 ships
  `py3-none` wheels for macOS arm64 and Linux aarch64, so any of these work.
  Use the `.venv` in the project root (`python3.13 -m venv .venv`); it is
  git-ignored and rebuilt on the Pi by `scripts/pi_setup.sh`.
- **Vision stack:** `opencv-python` for capture, compositing and display;
  MediaPipe Image Segmenter (selfie / person model) as the primary person
  segmentation engine; classical frame‑differencing against the held frame as a
  cheap fallback/refinement.
- **Camera:** enumerate devices at startup. On Linux use V4L2 and prefer devices
  whose sysfs path is under a USB bus; on macOS use AVFoundation and probe
  indices. Never assume index 0.
- **Performance target:** ≥ 15 FPS on Raspberry Pi 4/5 at 640×480 processing
  resolution (display can be upscaled). Segmentation runs at reduced
  resolution; masks are upscaled and feathered.
- **Configuration:** all tunables (camera preference, resolution, alpha, glow
  colour/thickness, model path, fullscreen, key bindings) live in a single
  `config.yaml`/`config.py`, not scattered through code.
- **No GUI framework beyond OpenCV** (`cv2.imshow` / fullscreen window). The Pi
  runs this as a kiosk via systemd.

## Repository layout

```
Ghost_Image/
├── AGENTS.md              # this file
├── README.md              # setup, operation, troubleshooting (Mac and Pi)
├── pyproject.toml         # project metadata, ruff/black/pytest settings
├── docs/
│   ├── PLAN.md            # step‑by‑step development plan
│   └── BENCHMARKS.md      # measured segmenter FPS
├── ghost_image/           # application package
│   ├── __main__.py        # entry point: python -m ghost_image
│   ├── app.py             # main loop / hold state
│   ├── camera.py          # USB camera discovery + capture
│   ├── segmentation.py    # MediaPipe, diff, and hybrid masks
│   ├── compositor.py      # ghost blend + contour glow
│   ├── ui.py              # window, overlays, keys
│   ├── config.py          # load/validate config
│   ├── gpio.py            # optional Pi buttons
│   ├── store.py           # held frame and snapshots
│   └── log.py             # rotating log + stage timings
├── models/                # downloaded .tflite models (git‑ignored)
├── scripts/               # pi_setup, model download, kiosk service
├── tests/                 # pytest; never requires a camera
├── config.yaml            # all tunables, documented inline
├── config.local.yaml      # optional per‑machine overrides (git‑ignored)
└── requirements.txt       # pinned deps with Mac arm64 + Linux aarch64 wheels
```

## Conventions for agents working here

- Follow `docs/PLAN.md`. Work one step at a time; do not skip ahead or
  implement later phases while working on an earlier one unless asked.
- Keep platform‑specific code inside `camera.py` (and, if needed, `ui.py`).
  Everything else must be pure OpenCV/NumPy and platform agnostic.
- Prefer NumPy/OpenCV vectorised operations; never loop over pixels in Python.
- Every processing stage should be individually toggleable/visualisable via a
  debug view (raw feed, mask, held frame, final composite) to make tuning on
  the Pi possible without a debugger.
- Do not commit model binaries or large media. Provide a download script.
- Keep dependencies minimal and pinned in `requirements.txt`; every new
  dependency must have a known aarch64 Linux wheel or an apt alternative.
- Run `pytest` before finishing a step. Tests must not require a camera; use
  synthetic frames.
- Use type hints and short docstrings. Format with `black`, lint with `ruff`.

## Operator controls (planned defaults)

| Key     | Action                                   |
|---------|------------------------------------------|
| `SPACE` | Hold current frame as background         |
| `R`     | Release held frame (back to live view)   |
| `D`     | Cycle debug views                        |
| `F`     | Toggle fullscreen                        |
| `[`/`]` | Decrease / increase ghost opacity        |
| `Q`/Esc | Quit                                     |

## Deployment notes

- Target: Raspberry Pi 4 or 5, 64‑bit Raspberry Pi OS (Bookworm), USB webcam.
- Development: macOS, same codebase, `python -m ghost_image`.
- The Pi runs the app at boot via a systemd user service in fullscreen; the
  service restarts on failure and the app re‑discovers the camera if it drops.
