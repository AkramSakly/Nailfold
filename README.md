<div align="center">

# Nailfold Capillaroscopy: Segmentation and Classification

**Deep learning (ResNet, PyTorch) for automatic analysis of nailfold capillaroscopy images**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Code-upload%20in%20progress-lightgrey)](#status)

</div>

---

## Overview

Nailfold capillaroscopy is a non-invasive imaging technique used to examine the small blood vessels (capillaries) at the base of the fingernail. Changes in capillary shape and density are early markers of microvascular diseases such as systemic sclerosis.

This project, carried out during my Research Master's in Computer Vision (Université de Bourgogne, 2025–2026), builds deep learning models that:

1. **Segment** capillaries in capillaroscopy images (pixel-wise masks).
2. **Classify** images into diagnostic categories.

## Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Capillary segmentation | ResNet-based encoder (PyTorch) | Dice | **0.85** |
| Capillary segmentation | ResNet-based encoder (PyTorch) | IoU | **0.74** |

## Methods

- **Backbone**: ResNet convolutional networks in PyTorch, used for both the classifier and as the encoder of the segmentation network.
- **Segmentation metrics**: Dice coefficient and Intersection over Union (IoU) on held-out images.

## Status

The training code, evaluation scripts and example predictions are being cleaned up and will be added to this repository. Contact me if you would like to discuss the project in the meantime.

## Author

**Akram Sakly**, Research Master's in Computer Vision, Université de Bourgogne (Le Creusot, France)
[GitHub](https://github.com/AkramSakly) · [LinkedIn](https://www.linkedin.com/in/akram-sakly)
