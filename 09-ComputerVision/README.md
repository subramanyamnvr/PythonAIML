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

## Notes

- Each top-level notebook starts with a small setup cell that switches into its original lesson folder before execution.
- The flat notebook copies had the branded logo header removed where it appeared as the first lesson cell.
- `39-Deep-Learning-Custom-Images.ipynb` expects a `DATA/CATS_DOGS` dataset, but that dataset is not bundled in this folder.
- `42-YOLO-Object-Detection.ipynb` points into the legacy YOLOv3 project, which is based on an older TensorFlow/Keras stack.
- Empty placeholder folders such as `05-Object-Tracking` and `07-Capstone-Project` can be added back later once real content exists.

## Support Files

- `DATA`
- `cvcourse_windows.yml`
- `cvcourse_linux.yml`
- `cvcourse_macos.yml`
- `06-Deep-Learning-Computer-Vision/06-YOLOv3`

## Good Next Additions

- Object tracking with KCF, CSRT, SORT, or DeepSORT
- Image segmentation
- OCR and document understanding
- Transfer learning with modern torchvision or timm models
- A real capstone project with evaluation, deployment, and error analysis
