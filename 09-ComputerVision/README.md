# Computer Vision Path

This folder now has a flat, numbered notebook layer at the top level.

The `DATA` folder and the legacy `06-Deep-Learning-Computer-Vision/06-YOLOv3` project stay in place because the lessons depend on those assets and local support files.

## Learning Order

- `01` to `04`: NumPy and image basics
- `05` to `10`: image basics with OpenCV
- `11` to `19`: image processing
- `20` to `24`: video basics
- `25` to `35`: classical object detection and feature extraction
- `36` to `41`: deep learning for computer vision with Keras
- `42`: YOLO object detection

## Expansion Folders

- `05-Object-Tracking`: starter space for KCF, CSRT, SORT, DeepSORT, ByteTrack-style pipelines, and tracking metrics
- `07-Image-Segmentation`: starter space for semantic, instance, and panoptic segmentation, plus mask metrics
- `08-OCR-and-Document-Understanding`: starter space for OCR, layout extraction, forms, receipts, and document QA
- `09-CV-Capstone`: starter space for a portfolio-ready computer vision project with evaluation, deployment, and error analysis

## Notes

- Each top-level notebook starts with a small setup cell that switches into its original lesson folder before execution.
- The flat notebook copies had the branded logo header removed where it appeared as the first lesson cell.
- `39-Deep-Learning-Custom-Images.ipynb` expects a `DATA/CATS_DOGS` dataset, but that dataset is not bundled in this folder.
- `42-YOLO-Object-Detection.ipynb` points into the legacy YOLOv3 project, which is based on an older TensorFlow/Keras stack.
- The numbered notebooks at the folder root are still the main path. The new expansion folders are where the next wave of CV topics should grow.

## Support Files

- `DATA`
- `cvcourse_windows.yml`
- `cvcourse_linux.yml`
- `cvcourse_macos.yml`
- `06-Deep-Learning-Computer-Vision/06-YOLOv3`

## Good Next Additions Inside The New Folders

- Tracking notebooks with detector-plus-tracker pipelines and MOT metrics
- Segmentation notebooks using U-Net, Mask R-CNN, SAM-style prompting, or modern segmentation backbones
- OCR notebooks covering classical OCR, layout parsing, and document QA
- One capstone that includes dataset versioning, error analysis, and a simple demo
