# CNN (Convolutional Neural Network)

here i implemented the famous architecture CNN.

it's a class that can be instatiated with different topologies and hyperparameters, so its possible to build a lot of architectures and test it out.

the dataset used is CIFAR-10.

this is a project of my master's degree classes with Prof. Dr. Lucas Ribas [].

## tests

here i tested 3 different archs, a small, a medium and a big one. but you can just modify the dict `architectures` and test how many you desire to.

## results

with some fine-tuning, changing some hyperparemters, regularizators, i reached 83% of accuracy with the medium architecture.

for more details, i wrote the `report.pdf`, using LaTeX, and you can read it to understand my choices and results obtained.

## plots

in the directory `output` there are some plost of feature maps, filters (kernels), history of acc and loss during epochs and some example of predictions done by the final model of each architecture.

## requirements

- torch
- torchvision
- matplotlib
- numpy