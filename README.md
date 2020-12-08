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

Simply run `train_light.py` and `train_temp.py`. The logs will be stored in `./logs` and the trained models will be stored in `./model`

## Generate test data

The test data for temperature is included in `./testdata/Test_set_temp`.

For light, run `Generate_testset_light.m`. 

## Testing the neural networks

The pretrained models are included in `./model` and you can directly run this part without re-training the neural networks

### Generate heatmaps

Run `test_light.py` and `test_temp.py`, and the heatmaps (confidence) will be stored in `./results`

### MLE results and Visualization

Run `MLE.m`. The visualization of heatmaps are stored in `./results/heatmap_visual`. 

Here is an example of visualization plot

![heatmap1](example_figs/4.png)

Here are the MLE localization results

![distribution](example_figs/pic1.png)
![localization](example_figs/pic2.png)
