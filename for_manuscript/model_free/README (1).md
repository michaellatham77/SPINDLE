# Comparing Modelfree Scripts
Scripts for generating and comparing Modelfree data to SPINDLE
Recommended to run on NMRPipe where Modelfree exits already.

## To Make Model 2 Data
To make NMR data for model 2 use generate_modelfree_model2.py3. This script makes 1000 random proteins that follow model 2 along with a text file and numpy file. 

### generate_modelfree_model2.py3 flags:

-h (help flag to print reminders of each flag)
  
-field (input the proton frequency in MHz, ex 800)*
  
-out (the name of the output directory being created containing the output txt and numpy file)*

*Needed for script to run

Example to run: ./generate_modelfree_model2.py3 -field 800 -out 800_md2

### Output of Script:

The resulting output is a numpy file (matrix containing created protein data) and text file (benchmark data) contains data for the resulting 1000 proteins that follow model 2. Both files will be used for downstream applications. 

## To Make Model 4 Data
To make NMR data for model 4 use generate_modelfree_model4.py3. This script makes 1000 random proteins that follow model 4 along with a text file and numpy file.

### Generate_modelfree_model4.py3 flags:

-h (help flag to print reminders of each flag)

-field (input the proton frequency in MHz, ex 800)*

-out (the name of the output directory being created containing the output txt and numpy file)*

*Needed for script to run

Example to run: ./generate_modelfree_model4.py3 -field 800 -out 800_md4Example to run: ./generate_modelfree_model4.py3 -field 800 -out 800_md4

### Output of Script: 

The resulting output is a new directory containing a numpy file and text file contains data for the resulting 1000 proteins that follow model 2. Both files will be used for downstream applications. 

## To Prepare the Data to Run for Modelfree
To prepare the Modelfree data to run for the parallel command for either model 2 or model 4 one will want to use either the prepare_modelfree_model2_run.py3 or prepare_modelfree_model4_run.py3 scripts. 

### For both prepare_modelfree_model2_run.py 3 or prepare_modelfree_model4_run.py3 scripts flags: 

-h (help flag to print reminders of each flag)

-field (input the proton frequency in MHz, ex 800)*
  
-benchmark_file (path to the text file made in previous scripts)*

-out_dir (path to directory made in previous scripts)*

*Needed for script to run

Example of running each script:
./prepare_modelfree_model2_run.py3 -field 600 -benchmark_file 600_md2/modelfree_model2_benchmark_data.txt -out_dir 600_md2
./prepare_modelfree_model4_run.py3 -field 800 -benchmark_file 800_md4/modelfree_model4_benchmark_data.txt -out_dir 800_md4

### Output of Script

The output assembles the previously made 1000 proteins into directories and input files that the Modelfree program will go through and read. Note, one must put the data in the previously created directory as this script will not create a new directory.


## Running Modelfree
The created and organized modelfree data was ran using the parallel command in NMRbox. This command forced GPU usage on NMRbox and ran modelfree using the data previously made in the steps above.

Example to run fast model free: time (find fmf_model4_runs -mindepth 1 -maxdepth 1 -type d -name "protein_*" | parallel --joblog stats_fmf_model4.txt --bar "cd {} && timeout 180m fastMF > fastMF.log 2>&1")

Example to run model 2 or model 4: time (find modelfree_model4_runs -mindepth 1 -maxdepth 1 -type d -name "protein_*" | ~/.local/bin/parallel --joblog stats_model4.txt --bar "cd {} && timeout 45m modelfree4 -i mfinput -d mfdata -p mfparam -m mfmodel -o mfout > mfrun.log 2>&1")

To run model 2 or another model one must replace after the find function the name of the directory containing the created proteins. One can also change the timeout point in minutes after timeout (how long the program runs until either it reaches this time or completion of the job before this time). 
The output to these command lines includes the filling every directory with output files from modelfree that reflect the progress of program and final answers if the program completed in time. Additionally, printed to the screen will be the real time it took to run, time it took if the parallel command wasn’t used, and time for the system in minutes and seconds. 

## Analyzing the Results of Modelfree
There are separate scripts to compare how well Modelfree (for model 2, model 4, or fast modelfree) fit to the data and how well SPINDLE fit the data which will be described in this section.

### To analyze how well Modelfree analyzed the data three scripts were used:

updated_analyze_modelfree_model2_results.py3

updated_analyze_modelfree_model4_results.py3

updated_FastModelfree_analyze.py3

These scripts analyze how well modelfree fit by comparing the data answers (known and created in the previous steps) to those from modelfree. These scripts present the data in pdf plots (used in the paper), csv files, and prints information about the runs to the screen. 

#### Flags for updated_analyze_modelfree_model2_results.py3 and updated_analyze_modelfree_model4_results.py3:

-h (help flag to print reminders of each flag)

-mf_dir (location of the directory of modelfree proteins)*

-data_file (location of the numpy file that has the ground truths)*

-cvs_outfile (name of the csv file being output)

-plot_file (name and location of the pdf for the correlation scatter plot that is being output)

-status_plot_file (name and location of pdf for the success/failure bar chart that is being output)

--plot (generates both possible plots)

*Needed for script to run

Note: If one wants the csv file then that flag can additionally be filled out. For either pdf files to be made then the –plot flag must be used but do not follow this flag with any other additional information. The –plot flag with also automatically generate both pdf files and give them automatic names that will be put in the same directory that script is run in unless specified by using the -plot_file and -status_plot_file flags. The pdf’s will not be generated if one only uses the -plot_file and/or -status_plot_file flags.

Example to run: ./updated_analyze_modelfree_model4_results.py3 -mf_dir 800_md4/fmf_model4_runs -data_file 800_md4/modelfree_model4_dataset.npz –plot -plot_file 800_md4/800_modelfree_4_scatter.pdf -status_plot_file 800_md4/800_modelfree_4_bar.pdf

#### Output of Scripts

The script will automatically print total RMSE values for each parameter and information about how many runs were completed. When the plot flag is used two pdf’s are generated: one with a scatter plot containing a plot for each parameter measuring the correlation between the ground truth and model free’s answer for each run that succeeded and a bar chart showing how many runs completed and didn’t. If selected a csv file containing information about what was extracted to represent modelfree will be shown. 

#### Flags for updated_FastModelFree_analyze.py3

-h (help flag to print reminders of each flag)

-mf_dir (location of the directory of modelfree proteins)*

-data_file (location of the numpy file that has the ground truths)*
  
-model_path (location of SPINDLE models)*

-out_prefix (prefix that will start all the pdfs that will be output)

*Needed for script to run

Note: The -out_prefix is optional and used to just add keywords to the pdfs that are automatically made. Finally, the script needs to be run in the same environment as SPINDLE (virtual environment, etc.) so that the models of SPINDLE load.

Example to run: ./updated_FastModelFree_analyze.py3 -mf_dir 800_md4/fmf_models_runs -data_file 800_md4/modelfree_model4_dataset.npz -model_path ../../models -out_prefix 800_fmf

#### Output of Script

The script will automatically print total RMSE values for each parameter and information about how many runs were completed. The script automatically produces pdfs containing a plot, a heatmap, and bar charts. The overall plot generated shows a plot for each parameter measuring the correlation between the ground truth and model free’s answer for each run that succeeded. The heatmap created shows how many residues were fit to which models and the ground truths in comparison. The two bar charts show in one how many residues were fit and excluded and the other shows how many residues were fit to which models.

## Analyzing the Results of SPINDLE

The following scripts are meant to analyze how well SPINDLE predicted the data versus the ground truth generated for model 2 or model 4.
To analyze how well SPINDLE predicted the data two scripts were used depending on the model:

updated_benchmark_spindle_model2.py3

updated_benchmark_spindle_model4.py3

These scripts analyze how well modelfree fit by comparing the data answers (known and created in the previous steps) to those from modelfree. These scripts present the data in pdf plots (used in the paper), csv files, and prints information about the runs to the screen. 

### Flags for both scripts:

-h (help flag to print reminders of each flag)

-model (location of SPINDLE models)*

-data_file (location of the numpy file that has the ground truths)*

--plot (generates both possible plots)

-plot_file (name and location of the pdf for the correlation scatter plot that is being output)

*Needed for script to run

Note: If one wants to have the pdf of the scatter plots made then both the --plot and -plot_file flags must be used. The plot flag must be used to obtain the scatter pdf the -plot_file flag is only used to direct the location of the created scatter and give a unique name. 

Example to run: ./updated_benchmark_spindle_model4.py3 -model ../../models -data_file 800_md4/modelfree_model4_dataset.npz --plot -plot_file 800_md4/SPINDLE_vs_model4_800.pdf

### Output of Script

The script will automatically print total RMSE values for each parameter and information about how many runs were completed. Additionally if the plot flag chosen then a scatter plot for each parameter measuring the correlation between the ground truth and SPINDLE’s answer for each run. 
