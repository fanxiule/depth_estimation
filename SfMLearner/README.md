## SfMLearner
This implementation is based on [tinghuiz/SfMLearner](https://github.com/tinghuiz/SfMLearner.git) 

Additional code is added to preprocess, train and test the NYU Depth v2 Dataset.
This package was tested on TensorFlow 1.13.

### Preprocess NYU Depth Data
Download the NYU raw data and put them in a folder in the following format:
```
/
    ../bedroom_0001/
        data...
    ../bedroom_0002/
        data...
    ...
```

Update where to find the above raw data and where to store the processed data 
in `./data/prepare_train_data_NYU.py`. Then run the script to prepare the data:
```
python ./data/prepare_train_data_NYU.py
```

### Train the network
First run the following command to download the checkpoint for the model trained on 
KITTI:
```
bash ./models/download_depth_model.sh
```

Update the file paths (where the processed data is, there the initial checkpoint is, 
where to store new checkpoints) in `train_nyu.py`. Then run
```
python train_nyu.py
```

### Test the network
Download the labeled dataset from NYU Depth v2 website and store the 
`nyu_depth_v2_labeled.mat` file in `./nyu_test`. Run the following 
the convert MATLAB information into actual images:
```
cd nyu_test
python pre_processed_NYU_test.py
```
Select the checkpoint you want to use for testing in `test_nyu.py` 
and run the script to get test results.
```
python test_nyu.py
```