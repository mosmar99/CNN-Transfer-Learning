# Transfer Learning in Convolutional Neural Networks

This project explores transfer learning for image classification using convolutional neural networks. The goal is to study how pretrained weights influence performance when training on a different dataset.

## Overview

Two datasets are used for the experiments:

- Stanford Dogs dataset with 20580 labeled images across 120 breeds  
- Cats vs Dogs dataset with 23412 valid images in two classes  

Both datasets are split into training and validation sets with a 20 percent validation split.

## Model Design

A pretrained CNN model with about 2.7 million parameters is used as the base model. The learning rate is set to 1e-4 and the batch size to 64. Only the learning rate and batch size were altered from the original implementation. The model is implemented in TensorFlow.

## Experiment Setup

The base model is trained first on the Stanford Dogs dataset. These weights are then transferred to models trained on the Cats vs Dogs dataset. A separate baseline model with random initialization is also trained for comparison.

Each experiment modifies which CNN layers are reinitialized:

- Only output layer replaced  
- First three convolutional layers replaced plus output layer  
- Last two convolutional layers replaced plus output layer  

For each configuration, two models are trained: one with transferred weights frozen and one with transferred weights trainable.

## Insights

Models using pretrained weights show significantly better initial performance compared to the baseline. The best performing models are those where the last two convolutional layers and the output layer are reinitialized. These models achieve the highest test accuracy in both frozen and unfrozen settings.

Freezing weights provides a regularizing effect by reducing the number of trainable parameters. This stabilizes the training process and lowers the risk of diverging updates. When weights are unfrozen, large gradients in early epochs can disrupt the structure learned in the pretrained model.

Overall, adding capacity near the end of the model yields better results than reinitializing early layers. Early layers contain general features that transfer well across datasets.

## Final Performance Summary

Best test accuracies by configuration:

- Baseline (no transfer): 0.9178  
- Only output replaced: 0.9244  
- First three conv layers replaced: 0.9180  
- Last two conv layers replaced: 0.9297  

The highest accuracy is reached by the model with the last two convolutional layers and the output reinitialized.

## Conclusion

Transfer learning provides a clear advantage in both starting accuracy and final performance. Preserving the early feature extraction layers while adapting the later layers leads to the best results. Freezing transferred weights aids stability while unfreezing them allows the model to achieve higher final accuracy when training conditions are well selected.

## Authors

- Mahmut Osmanovic  
- Isac Paulsson
