# Deep Learning Module Map

This map shows the new numbered sequence and the intent of each folder.

## Foundations

- `01-Regression-Analysis`  
  Formerly `Regression Analysis`. Linear regression, multiple regression, gradient descent, polynomial regression, regularization, cross-validation, and grid search.

- `02-Logistic-Regression`  
  Formerly `Logistic Regression`. Probabilities, metrics, cross-validation, multiclass work, and a challenging dataset pass.

- `03-Neural-Networks-Regression`  
  Formerly `Neural Networks for Regression`. Dense-network regression basics in the core notebook set.

- `04-Neural-Networks-Regression-TensorFlow`  
  Formerly `Neural Networks for Regression_TF`. TensorFlow/Keras versions of the regression path.

- `05-Neural-Networks-Classification`  
  Formerly `Neural Networks for Classification`. Dense-network classification, MNIST, custom-image classification, and HAR examples.

- `06-Neural-Networks-Classification-TensorFlow`  
  Formerly `Neural Networks for Classification_TF`. TensorFlow/Keras classification notebooks with MNIST and custom-image examples.

- `07-Custom-Layers`  
  Formerly `Creating Custom Layer`. Custom layer construction in PyTorch and TensorFlow.

- `08-Custom-Loss-Functions`  
  Formerly `Create Custom Loss Function`. Custom loss design in PyTorch and TensorFlow.

## Optimization and Regularization

- `09-Learning-Rate-Scheduling`  
  Formerly `Scheduling Learning Rate`. Learning-rate schedules in PyTorch and TensorFlow.

- `10-Dropout-and-Batch-Normalization`  
  Formerly `Dropout Regularization and Batch Normalization`. Regularization and stabilization techniques for tabular, image, and wearable-sensor inputs.

- `11-Early-Stopping`  
  Formerly `Early Stopping Criterion for NN`. Early stopping examples in PyTorch and TensorFlow.

## Vision and Spatial Models

- `12-Convolutional-Neural-Networks`  
  Formerly `Convolutional Neural Network (CNN)`. Shapes, MNIST, custom images, and transfer learning with ResNet/VGG-style examples.

- `13-Convolutional-Neural-Networks-TensorFlow`  
  Formerly `CNN_TF`. TensorFlow/Keras CNN path with custom-image and transfer-learning notebooks.

- `14-One-Dimensional-CNN-PyTorch`  
  Formerly `One Dimensional CNN Pytorch`. One-dimensional CNNs for shapes and time-series classification.

- `15-One-Dimensional-CNN-TensorFlow`  
  Formerly `One Dimensional CNN Tensor Flow`. TensorFlow/Keras one-dimensional CNN notebooks.

## Sequence Models

- `16-Recurrent-Neural-Networks`  
  Formerly `Recurrent Neural Network (RNN)`. LSTM sequence modeling, interpolation, and classification on sequence data.

- `17-LSTM-TensorFlow`  
  Formerly `LSTM_TF`. TensorFlow/Keras LSTM notebooks including prediction and text-classification examples.

- `18-Bidirectional-LSTM-PyTorch`  
  Formerly `Bidirectional LSTM Pytorch`. Bidirectional LSTM examples in PyTorch.

- `19-Bidirectional-LSTM-TensorFlow`  
  Formerly `Bidirectional LSTM TF`. TensorFlow/Keras bidirectional LSTM notebooks.

## Autoencoders and Generative Models

- `20-Autoencoders`  
  Formerly `Autoencoders`. Denoising, occlusion removal, and classifier-style autoencoder experiments.

- `21-CNN-Autoencoders`  
  Formerly `CNN Autoencoder`. Convolutional encoder-decoder models and classification-oriented variants.

- `22-LSTM-Autoencoders`  
  Formerly `LSTM Autoencoder`. Sequence autoencoders for shapes and waveform reconstruction.

- `23-Variational-Autoencoders`  
  Formerly `Variational Autoencoder`. Fully connected and convolutional VAE examples.

- `24-Generative-Adversarial-Networks`  
  Formerly `Generative Adversarial Network (GAN)`. Core GAN training on MNIST.

- `25-Deep-Convolutional-GANs`  
  Formerly `Deep Convolutional (DC) GAN`. DCGAN work for MNIST and image-generation tasks.

- `26-Neural-Style-Transfer`  
  Formerly `Neural Style Transfer`. VGG/AlexNet style-transfer notebooks with local image assets and helper code.

## Transformers

- `27-Transfer-Learning-NLP-Transformers`  
  Formerly `Transfer Learning NLP Transformer`. Sentiment, text generation, masked language modeling, summarization, translation, and QA.

- `28-Fine-Tuning-NLP-Transformers`  
  Formerly `Fine Tuning NLP Transformer`. Tokenization and fine-tuning workflow for transformer-based NLP.

- `29-Time-Series-Transformers`  
  Formerly `Time Series Transformer`. Transformer encoders and sequence modeling for time-series reconstruction and classification.

- `30-Vision-Transformers`  
  Formerly `Vision Transformer`. Vision-transformer fine-tuning and masked-autoencoder style notebooks.

## Reinforcement Learning

- `31-Reinforcement-Learning-Basics`  
  Starter expansion folder for bandits, Markov decision processes, Q-learning, policy gradients, and environment-driven experimentation.

## Layout Notes

- Topic folders are now numbered continuously from `01` to `31`.
- Support files remain beside the notebooks that use them.
- The goal was to flatten the learning path, not to break image-folder datasets or sequence-data layouts that the notebooks depend on.
- Portable setup cells were added to the notebooks that previously depended on hardcoded Colab drive navigation.
