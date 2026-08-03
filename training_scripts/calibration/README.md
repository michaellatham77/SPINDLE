# Instructions for validation and calibration of machine learning models
This README walks through the process of validating and calibrating the trained models. Note, this process has to be done for each set of models that are generated.

## Generate test data 
Unless tensorflow and keras are natively in your python path, it's best to be in your virtual environment.
```bash source ~/spindle/bin/activate```
Then generate test dataset.
```bash
./generate_final_exam.py3 -field 500 \
  -n_proteins 10000 \
  -out_dir 500
```
This will generate two files in the directory ```500```: ```final_exam_dataset.npz``` and ```modelfree_benchmark_data.txt```.

## Run DNN on test data
```bash
./get_ensemble_predictions.py3 -model_dir ../../models/500 \
  -data_file 500/final_exam_dataset.npz \
  -out_file 500/final_exam_results.npz
```
This will generate one file in the directory ```500```: ```final_exam_results.npz```.

## Analyze the predictions and perform calibrations
```bash
./analyze_and_correct.py3 -results_file 500/final_exam_results.npz\
  -plot_file 500/final_performance_plot.png \
  -Rex_vs_size \ #Optional analysis of Rex precision and sensitivity vs $B_0$ as a function of $/tau_c$
  -error_analysis \ #Optional analysis of correlation of model error with true error
  -correct_error #Optional error correction and ambiguity threshold calibration
```
This will generate plots of the predicted vs ground truth dynamics parameters (```final_performance_plot.png```) and print various information out to the terminal window. 

Note, the performance plot for the 500 MHz validation data and the ```-Rex_vs_size``` flag was used to make Fig. 2. 
