# Ghost Image

An interactive art installation built with Python and OpenCV. A USB camera
watches a fixed scene; the operator **holds** a reference frame of that scene,
and from then on anyone who walks into view is rendered as a translucent,
edge-lit **ghost** floating over the held frame, with the original scene
visible through their body, hands and fingers.

Developed on macOS, deployed on a Raspberry Pi. Same code on both.

See [`docs/PLAN.md`](docs/PLAN.md) for the step-by-step development plan and
[`AGENTS.md`](AGENTS.md) for the project overview and conventions.

## Requirements

- Python **3.11 or newer** (3.13 recommended; it is what Homebrew and current
  Raspberry Pi OS ship).
- A USB webcam (the built-in camera works for development on a Mac).
- macOS (Apple Silicon) or 64-bit Raspberry Pi OS. All dependencies ship
  pre-built wheels for both, so nothing is compiled during install.

## Setup on macOS (development)

```bash
# Python 3.13 via Homebrew if you don't have it: brew install python@3.13
cd Ghost_Image
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Setup on Raspberry Pi (deployment)

The virtual environment is **not** copied from the Mac; it is rebuilt on the Pi
from `requirements.txt`.

```bash
git clone <your-repo-url> Ghost_Image
cd Ghost_Image
./scripts/pi_setup.sh          # apt packages + venv + pip install
source .venv/bin/activate
```

To update later: `git pull`, then `pip install -r requirements.txt` if the
requirements changed.

### Smoke test

After setup, confirm the USB camera and record a baseline (no window, no
segmentation). The FPS line is the number to keep when choosing a processing
resolution later:

```bash
./scripts/pi_smoke.sh
```

That lists the cameras, then reads for 5 seconds and prints `fps=` and `cpu=`.
A live window is the same command as on the Mac: `python -m ghost_image`.

## Running

```bash
source .venv/bin/activate
python -m ghost_image                # run the app
python -m ghost_image --version
python -m ghost_image --list-cameras # show detected cameras (USB preferred)
python -m ghost_image --show-config  # print the resolved configuration
python -m ghost_image -c other.yaml  # use a different config file
```

`python -m ghost_image` opens the camera in a window. Press `F` to toggle
fullscreen and `Q` or `Esc` to quit. The FPS is drawn in the corner.

## Configuration

All tunables live in [`config.yaml`](config.yaml) (camera selection,
resolution, ghost opacity, glow colour, key bindings, ...). Every key is
commented there.

For machine-specific overrides, create a `config.local.yaml` next to it with
only the keys you want to change. It is git-ignored and merged on top of
`config.yaml`. Example for a Pi kiosk:

```yaml
camera:
  width: 640
  height: 480
processing:
  width: 256
display:
  fullscreen: true
```

Unknown keys and out-of-range values are rejected at startup with a clear
error, so typos don't fail silently.

## Development

```bash
pytest            # unit tests (no camera needed)
ruff check .      # lint
black .           # format
```

## Project layout

```
ghost_image/      application package (python -m ghost_image)
  config.py       config loading + validation
docs/PLAN.md      development plan
scripts/          Pi setup / service install scripts
models/           downloaded segmentation models (git-ignored)
tests/            pytest suite
config.yaml       all tunables
requirements.txt  pinned dependencies (Mac + Pi)
```
