# Instructions for how to generate training data and train the models
This README walks through the process of training data generation and model training using the HTCondor compute machines on [NMRBox](https://nmrbox.nmrhub.org).

## Make training data
Because training data is stored in a series of tensor flow record files, you have to be in your python virtual environment.
```bash
source ~/spindle/bin/activate
./make_training_data.py3 -field 500 \
  -n_proteins 400000 \
  -out_dir ../training_data/500 \
  -n_files 40
```
This will make 15N $R_1$, $R_2$, heteronuclear NOE relaxation data for 400k proteins at 500 MHz, storing those in 40 files in a directory called ```../training_data/500```.

## Train models
Training the ensemble of models using HTCondor requires three files/scripts: ```run_condor_training.sh```, ```train_ensemble.sub```, ```train_model_condor.py3```.
1. Update ```run_condor_training.sh``` with correct python virtual environment path.
   * Note, this only has to be done once.
2. Update ```train_ensemble.sub``` with correct paths for training data, writing the models, and log files.
   * Note, 10 runs are submitted to the queue (```queue 10```) which will be the 10 models in the DNN ensemble.
3. Submit the job to HTCondor: ```condor_submit train_ensemble.sub```
   

When training is done, 10 models will be made in ```../models/500``` in this case. Training curves (global and local loss vs training epoch) will also be generated for each model.

Please see README.md in ```calibration``` for the outline of the process for validation and calibration of the resulting models.
