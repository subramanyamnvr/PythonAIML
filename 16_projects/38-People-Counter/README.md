# People Counter

## Overview
This project combines YOLO-based detection with SORT tracking to count people moving through a scene.

## Included Here
- `People-Counter.py`: the main inference and counting script
- `sort.py`: the tracking implementation used by the counter
- `mask.png` and `graphics.png`: supporting visual assets
- `requirements.txt`: the packages needed for the tracking demo

## Run
1. Install dependencies with `pip install -r requirements.txt`.
2. Review local file paths and model references inside `People-Counter.py`.
3. Run `python People-Counter.py`.

## Notes
- The script expects local video/model assets to be available, so keep those paths aligned with your machine.
- This README intentionally documents the current entrypoint without renaming the original script file.
