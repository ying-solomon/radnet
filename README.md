# RadNet
Deep neural network for climate science radiation functions

## Train
To train a model use train.py
Example:

```
python ./train.py --data_dir ./historical_dataset/ --test_dir ./historical_dataset/
```

In this case, we pass the data using the option "--data_dir" and test with "--test_dir". inside data are many folders each one containing samples of the dataset, so no matter how many nested dirs, the reader will reach the .nc files.
The model will generate a dir in which it places the models and the summaries of each model. The model will store only the latest models.


## TensorBoard
Just pass the directory that contains the models and log summaries.

```
tensorboard --logdir ./logdir/
```

## Project features
The model is able to save the model, load a model and keep training it even with other hyperparameters, log loss summaries tensorboard and tune other features. For more help: 

```
python train.py --help
```

## File loader
The fileReader.py includes various methods of loading the ERA-interium dataset, with/out interpolation/extrapolation
of a static pressure grid. And different ways of preparing ERA-interium dataset to be the input of the neural networks.
Make sure that the trained model takes in the same input structure when inferencing.

## Training and Inferencing
The file train.py generates when finished, or interrupted with ctrl+c generates a file called "graph_frozen_radnet.pb" (in logdir/train/<date>/graph_frozen_radnet.pb) that contains the final state of the model that can be loaded for inferencing the model. 

The file test_radnet_script.ipynb contains an example on how to use the inferencing library that builds up the Tensorflow graph into memory and can be fetched in further calls.
It also has various helper functions to calculate statistics of the input dataset and predictions.

## single column model
The file test_equilibrium.ipynb compares radnet radiation prediction with climt rrtmg radiation calculation. They
can also drive the single column model to equilibrium.