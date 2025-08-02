# OxStressNet: A Neural Network-Based Method for Quantitative Estimation of Oxidative Stress Levels
OxStressNet is a computational framework designed to quantitatively estimate oxidative stress activation across diverse biological contexts using transcriptomic data.
# 1. Overview
This repository contains the source code for the paper "OxStressNet: A Neural Network-Based Method for Quantitative Estimation of Oxidative Stress Levels." The directory is organized as follows:

code/01.OS.GSVA.score.R: Computes the initial oxidative stress-related scores using gene set variation analysis (GSVA), including
O: ROS production score
R: Antioxidant response score
OS: Net oxidative stress response and cellular damage score

code/02.neural.network.py: Implements the neural network model that takes the GSVA-derived O, R, and OS scores as input and learns to predict the final oxidative stress level under biologically constrained relationships.

