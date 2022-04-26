# Query Efficient Model Extraction with Self-Supervised Contrastive Learning

This repository contains the code and presentation materials for the final project for the 2022 Machine Learning and Optimization course at RPI. The goal of this project was to show how we can use self-supervised learning to enhance a extraction attack on a target model. This work is an extension of High "Accuracy and High Fidelity Extraction of Neural Networks" https://arxiv.org/abs/1909.01838. We demonstrate experiments using the ImageNet models from "Billion-scale Semi-Supervised Learning for Image Classification" https://arxiv.org/abs/1905.00546 as a target.

# Methodology
First use src/extract_oracle_logits.py to compute all output logits of the target model. We next obtain pretrained self-supervision models from https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md. The self-supervised models are then finetuned using src/train.py, and evaluated using src/test.py. See the presentation slides for the specific training protocols we follow for each experiment, and additional methodology specifics.
