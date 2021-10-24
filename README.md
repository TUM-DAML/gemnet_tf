# GemNet: Universal Directional Graph Neural Networks for Molecules

Reference implementation in TensorFlow 2 of the geometric message passing neural network (GemNet). You can find its [PyTorch implementation in another repository](https://github.com/TUM-DAML/gemnet_pytorch). GemNet is a model for predicting the overall energy and the forces acting on the atoms of a molecule. It was proposed in the paper:

**[GemNet: Universal Directional Graph Neural Networks for Molecules](https://www.in.tum.de/daml/gemnet/)**   
by Johannes Klicpera, Florian Becker, Stephan Günnemann   
Published at NeurIPS 2021.

## Run the code
Adjust the config.yaml and config_seml.yaml file to your needs.
This repository contains a notebook for training the model (`train.ipynb`) and for generating predictions on a molecule loaded from ase (`predict.ipynb`). It also contains a script for training the model on a cluster with Sacred and [SEML](https://github.com/TUM-DAML/seml) (`train_seml.py`).

## Derive scaling factors
You can either use the precomputed scaling_factors (in scaling_factors.json) or derive them yourself by running fit_scaling.py. Scaling factors are the same for all GemNet variants.

## Contact
Please contact klicpera@in.tum.de if you have any questions.

## Cite
Please cite our papers if you use the model or this code in your own work:

```
@inproceedings{klicpera_gemnet_2021,
  title = {GemNet: Universal Directional Graph Neural Networks for Molecules},
  author = {Klicpera, Johannes and Becker, Florian and G{\"u}nnemann, Stephan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}
```

