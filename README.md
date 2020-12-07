# Migrating Monarch Butterfly Localization UsingMulti-Modal Sensor Fusion Neural Networks

## Retrive raw data

Go to dataset folder, run 
```
cat raw_data.tar.bz2.parta* >backup.tar.bz2
tar -xvjf raw_data.tar.bz2
```

Then you are expected to get `raw_train_data.mat` and `raw_test_data.mat`

## Generate training data

The training data for temeprature is included as `./dataset/Temp_train_16.mat` and  `./dataset/Temp_valid_16.mat`

To generate the training data for light, you need to run the MATLAB script `Generate_trainset_light.m`. Then, you are expected to get `./dataset/Light_train_8.mat` and `./dataset/Light_valid_8.mat`.

## Train the neural networks

