# Ghost Image

An interactive art installation built with Python and OpenCV. A USB camera
watches a fixed scene. The operator **holds** a reference frame of that scene,
and from then on anyone who walks into view is rendered as a translucent,
edge-lit **ghost** over the held frame, with the original scene visible
through their body, hands and fingers.

Developed on macOS, deployed on a Raspberry Pi. Same code on both.

See [`AGENTS.md`](AGENTS.md) for project conventions and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
for measured frame rates.

## Requirements

- Python **3.11 or newer** (3.13 is what Homebrew and current Raspberry Pi OS ship).
- A USB webcam. The built-in camera works for development on a Mac.
- macOS (Apple Silicon) or **64-bit** Raspberry Pi OS. MediaPipe does not
  publish 32-bit ARM wheels.

## Setup on macOS

```bash
cd Ghost_Image
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
./scripts/download_models.sh
```

## Setup on Raspberry Pi

The virtual environment is rebuilt on the Pi. Do not copy `.venv` from the Mac.

```bash
git clone <your-repo-url> Ghost_Image
cd Ghost_Image
./scripts/pi_setup.sh          # apt packages, venv, pip, models
source .venv/bin/activate
./scripts/pi_smoke.sh          # list the USB camera and print baseline FPS
```

Later updates: `git pull`, then `pip install -r requirements.txt` if
requirements changed, and `./scripts/download_models.sh` if the model file is missing.

For a kiosk that starts at login, set `display.fullscreen: true` in
`config.local.yaml` and run `./scripts/install_service.sh`. The service waits
for the graphical session, turns off screen blanking when `xset` is available,
hides the cursor when `unclutter` is installed, and restarts if the app exits.
Logs: `journalctl --user -u ghost-image -f`.

## Running

```bash
source .venv/bin/activate
python -m ghost_image                 # open the camera window
python -m ghost_image --list-cameras  # show what was detected
python -m ghost_image --benchmark-seconds 5   # FPS with no window
python -m ghost_image --show-config
python -m ghost_image --no-restore-held       # do not reload captures/held.png
```

### Keys

| Key | Action |
|-----|--------|
| Space | Hold the current frame (averaged over `camera.hold_frames`) |
| R | Release it and delete the saved reference |
| D | Cycle the view: composite, live, mask, held frame |
| F | Fullscreen |
| [ / ] | Ghost more transparent / more solid |
| S | Save a snapshot of the current view to `captures/` |
| Q or Esc | Quit |

The corner shows `LIVE` or `HELD`, the FPS, opacity, glow, segmenter, and the
current view. In the mask / held / live views it also shows how long
segmentation and compositing took.

On a Pi, two GPIO buttons can do the same job as Space and R. Set
`gpio.enabled: true` in `config.local.yaml` and wire BCM pins 17 (hold) and
27 (release), or change `gpio.hold_pin` / `gpio.release_pin`. This uses
`gpiozero`, which Raspberry Pi OS already provides. It is not installed on
the Mac, and the app keeps running from the keyboard if the package is missing.

The held frame is saved to `captures/held.png` and loaded on the next launch,
so a power cut does not forget the reference.

## Configuration

Every tunable is in [`config.yaml`](config.yaml), with a comment on each key.
Machine-specific overrides go in `config.local.yaml` (git-ignored), for example
a Pi kiosk:

```yaml
camera:
  width: 640
  height: 480
processing:
  width: 256
display:
  fullscreen: true
```

Unknown keys and out-of-range values are rejected at startup.

`processing.segmenter` is `mediapipe` (default), `diff` (no model, compares
the live frame with the held one), or `hybrid` (the model plus nearby motion,
which helps thin fingers). `processing.frame_skip` reuses a mask for extra
frames when the Pi is short of time.

## Troubleshooting

**No cameras found.** On the Pi, run `v4l2-ctl --list-devices` and
`python -m ghost_image --list-cameras`. Only nodes that advertise video
capture and sit on a USB bus are used. Set `camera.preferred_index` or
`camera.preferred_name` if the wrong device is chosen. On a Mac, index 0 is
treated as the built-in camera; plug in the USB webcam or set
`camera.preferred_index`.

**The window says the camera was lost.** Unplug and replug the USB camera.
The app rediscovers it after `camera.reconnect_failures` bad reads. It does
not exit.

**Model not found.** Run `./scripts/download_models.sh`. The files live in
`models/` and are not in git. If MediaPipe still cannot start, the app prints
one error and falls back to the `diff` segmenter.

**Low frame rate.** Read [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and run
`python scripts/bench_segmenters.py` on the Pi. The target is at least 15 FPS.
Lower `processing.width`, raise `processing.frame_skip`, or set
`processing.segmenter: diff`. Capture size is `camera.width` / `camera.height`;
segmentation is a separate, smaller width.

**The ghost outline flickers or misses fingers.** Raise `processing.mask_smoothing`,
increase `processing.morph_px`, or switch to `hybrid`. `processing.background_adapt`
lets the diff mask absorb slow lighting changes without changing the picture
the audience sees.

## Development

```bash
source .venv/bin/activate
pytest
ruff check .
black --check .
```

Tests use synthetic frames. They do not open a camera or a window.
