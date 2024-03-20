# My adapted implementation of SymmetryNet

## About

This is an implementation of an adaptation of SymmetryNet that focuses on
the prediction of planar reflective symmetries. It doesnt use the dense prediction scheme
described in the original work.

The final implementation used is the one of the center_n_normals_net. It can be used
using the torch lighting cli interface. The simple_net also works.

The implementation contained in the symmetry_net file is currently broken due the implementation
of the loss function. The results presented in my final work are from a center_n_normals_net that predicted
the test dataset and was evaluated using the file evaluation.py from scripts. Here is the link to a folder that contains the models:

https://drive.google.com/drive/folders/1vbFDW9e6LS5pc-Ev_SLUHe8zj1TpFptW?usp=drive_link

## A rough resume of the folder structure

config -> Holds the config files used to create models.

notebooks -> Just a simple notebook exploring the obtained results.

scripts -> Holds different utilities to explore and evaluate the model.

src -> Holds the main code of the project order by:

    - dataset -> Holds the Pytorch and Pytorch Lighting code to create the dataset and the datamodule.
    - metrics -> Holds my code to calculate the metrics of the models while training. WARNING this code does not seems to match the metrics calculated by the evaluation script.
    - model   -> the model definition ordered in different small parts.
    - utils   -> Common utils folder where i put thing i didnt know where to put.
    
My email is gsantelicesn2 (at) gmail (dot) com Ask freely!



