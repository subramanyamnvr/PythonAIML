# Deep Learning Path

This folder is now organized as one flat, numbered topic folder per module.

- The first layer is the learning path.
- Datasets, image-class folders, outputs, and helper scripts stay inside the topic folders that need them.
- Many notebooks had hardcoded Colab `drive.mount` and `%cd` cells. Those were replaced with a small portable setup cell so the notebooks are easier to run from the repo.

Open [module_map.md](module_map.md) for the detailed folder-by-folder breakdown, or open [learning_path.html](learning_path.html) for a visual view of the sequence.

## Recommended Order

### Stage 1: Foundations

- `01-Regression-Analysis`
- `02-Logistic-Regression`
- `03-Neural-Networks-Regression`
- `04-Neural-Networks-Regression-TensorFlow`
- `05-Neural-Networks-Classification`
- `06-Neural-Networks-Classification-TensorFlow`
- `07-Custom-Layers`
- `08-Custom-Loss-Functions`

### Stage 2: Optimization and Regularization

- `09-Learning-Rate-Scheduling`
- `10-Dropout-and-Batch-Normalization`
- `11-Early-Stopping`

### Stage 3: Vision and Spatial Models

- `12-Convolutional-Neural-Networks`
- `13-Convolutional-Neural-Networks-TensorFlow`
- `14-One-Dimensional-CNN-PyTorch`
- `15-One-Dimensional-CNN-TensorFlow`

### Stage 4: Sequence Models

- `16-Recurrent-Neural-Networks`
- `17-LSTM-TensorFlow`
- `18-Bidirectional-LSTM-PyTorch`
- `19-Bidirectional-LSTM-TensorFlow`

### Stage 5: Autoencoders and Generative Models

- `20-Autoencoders`
- `21-CNN-Autoencoders`
- `22-LSTM-Autoencoders`
- `23-Variational-Autoencoders`
- `24-Generative-Adversarial-Networks`
- `25-Deep-Convolutional-GANs`
- `26-Neural-Style-Transfer`

### Stage 6: Transformers

- `27-Transfer-Learning-NLP-Transformers`
- `28-Fine-Tuning-NLP-Transformers`
- `29-Time-Series-Transformers`
- `30-Vision-Transformers`

### Stage 7: Reinforcement Learning

- `31-Reinforcement-Learning-Basics`

## Practical Notes

- Start with the lowest-numbered folder in a stage, then move upward.
- Within each folder, follow the notebook filenames in `Project_01`, `Project_02`, `Project_03` order where available.
- Folders that use `Circle_Cross`, `CatDog`, `WESAD_data`, or similar dataset trees intentionally keep those nested directories because the notebooks expect class-based image or sequence inputs.
- The deep learning folder is flat at the topic level now; no extra curriculum nesting is needed to find the next module.
- `31-Reinforcement-Learning-Basics` is a starter expansion folder for adding tabular RL, policy gradients, and environment-centric notes without overloading the transformer section.
