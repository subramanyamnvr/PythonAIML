# Deep Learning Libraries Path

This folder now has a flat, numbered notebook layer at the top level.

The original `pytorch-deep-learning-main` and `tensorflow-deep-learning-main` folders stay here because many lessons depend on their helper modules, datasets, images, models, and script-based exercises.

## Learning Order

1. `01-PyTorch-Fundamentals.ipynb`
2. `02-PyTorch-Workflow.ipynb`
3. `03-PyTorch-Classification.ipynb`
4. `04-PyTorch-Computer-Vision.ipynb`
5. `05-PyTorch-Custom-Datasets.ipynb`
6. `06-PyTorch-Going-Modular-Cell-Mode.ipynb`
7. `07-PyTorch-Going-Modular-Script-Mode.ipynb`
8. `08-PyTorch-Transfer-Learning.ipynb`
9. `09-PyTorch-Experiment-Tracking.ipynb`
10. `10-PyTorch-Paper-Replicating.ipynb`
11. `11-PyTorch-Model-Deployment.ipynb`
12. `12-TensorFlow-Fundamentals.ipynb`
13. `13-TensorFlow-Neural-Network-Regression.ipynb`
14. `14-TensorFlow-Neural-Network-Classification.ipynb`
15. `15-TensorFlow-Convolutional-Neural-Networks.ipynb`
16. `16-TensorFlow-Transfer-Learning-Part-1-Feature-Extraction.ipynb`
17. `17-TensorFlow-Transfer-Learning-Part-2-Fine-Tuning.ipynb`
18. `18-TensorFlow-Transfer-Learning-Part-3-Scaling-Up.ipynb`
19. `19-TensorFlow-Food-Vision-Milestone-Project-1.ipynb`
20. `20-TensorFlow-Introduction-to-NLP.ipynb`
21. `21-TensorFlow-SkimLit-NLP-Milestone-Project-2.ipynb`
22. `22-TensorFlow-Time-Series-Forecasting.ipynb`

## Notes

- Each top-level notebook starts with a small setup cell that switches into its original lesson folder before execution.
- The PyTorch "going modular" section is included as two flat notebook entry points because that lesson mixes notebook work with reusable training scripts.
- Some notebooks download datasets or helper files on first run, while others rely on assets already present in the source repo folders.
- The original slides, extras, docs, and helper modules remain inside the source repo folders.

## Useful Source Areas

- `pytorch-deep-learning-main/going_modular`
- `pytorch-deep-learning-main/data`
- `pytorch-deep-learning-main/helper_functions.py`
- `tensorflow-deep-learning-main/extras`
- `tensorflow-deep-learning-main/video_notebooks`

## Good Next Additions

- PyTorch Lightning or Fabric
- Hugging Face transformers for text and vision
- ONNX or TorchScript export and inference optimization
- Mixed precision and distributed training
- Model serving with FastAPI, BentoML, or similar tooling
