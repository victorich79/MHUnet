# MHUNet: Deep Learning-Based Hemodynamic Prediction

This repository provides the implementation of **MHUNet**, a deep learning surrogate model developed for predicting hemodynamic indicators in abdominal aortic aneurysms (AAA).

The model predicts wall shear stress–related hemodynamic quantities directly from geometry-derived images without performing computational fluid dynamics (CFD) simulations.

## Features

- Multi-view geometry representation
- Deep learning–based hemodynamic prediction
- Fast alternative to CFD simulations
- Prediction and evaluation metrics

## Model

The implemented model is **MHUNet**, an extension of the MultiViewUNet architecture designed to predict hemodynamic maps from geometry-based inputs.

Input:
- Curvature images derived from AAA geometry

Output:
- Hemodynamic indicators (e.g., TAWSS)
