# BMRB Scripts
Scripts for reading through BMRB starfiles and making related figures.

## To Run Through Zip or TGZ files

For going through a bunch of zip files or the tgz files (zip format BMRB compiles everything in) use spindle_pipline_9.py3. This script will use the SPINDLE models, the zip files, and ignore any specific files. The output will be a directory containing pdf files and csv files of every condition/field of every starfile and the average of the conditions/field, the global statistics and comparison in pdf and csv files, and csv file with DOI’s that could be determined. As the script runs an output will appear on the screen on if a starfile will be skipped due to compatibility issues with SPINDLE and if any units need to be changed to follow SPINDLE’s units. 

### Spindle_pipline_9.py3 flags: 

-h (help flag to print reminders of each flag)

-in (put in the location of the zip or tgz files)* 

-out (the name of the output directory being created containing the output pdf and csv files)* 

-models (location of SPINDLE model’s directory)* 

-ignore (the BMRB entry code of any starfiles to be skipped in the zip which are separated by a single space)

*Needed for script to run

Ex.) ./spindle_pipline_9.py3 -in bmrb_entries.tgz -out bmrb_output -models ../../models -ignore 4970 51418

Note: Spindle_calibrations.py must be in the same directory to run, as this script contains the calibrations needed for the models to run correctly. Finally, the script needs to be run in the same environment as SPINDLE (virtual environment, etc.) so that the models of SPINDLE load.

### Output of Script

The resulting pdf files will contain graphs for each independent condition, the average of the conditions/fields, and global fit of the files together showing the fitting of spindle vs the BMRB file for the order parameter, &Tau;<sub>e</sub>, and R<sub>ex</sub> if the BMRB file had R<sub>ex</sub> data. The resulting number of residues fit for each, Pearsons correlation, RMSE, and MAE for each graph will be shown. For the independent files and the average of these files the predicted global tumbling time will be reported along with the condition/field and BMRB starfile entry number as the main header. The csv files will contain the residue number; S<sub>2</sub> and TauE reported from the BMRB; residue; S<sub>2</sub>, S<sub>2</sub> error, TauE, TauE error, R<sub>ex</sub>, and R<sub>ex</sub> error predicted from SPINDLE; the quality of fitting; TauC reported by SPINDLE; the field; and the condition.

The DOI csv will automatically be made and contain any DOI’s if they are in the BMRB starfile and report them for the correlating starfile. 
The Global_S2_pearson_histogram.pdf will automatically be made and contain a histogram showing the spread of how S<sub>2</sub> is being fit to Pearsons correlation across the whole dataset. 

## To Look at Independent Starfiles

To look at independent starfiles and obtain the figures seen in the paper that compare the BMRB data to SPINDLE’s output residue by residue one must use the spindle_single_pipleine_2.py3 script. The script can read in independent starfiles or full tgz with specific targets. The output is a directory containing .pdf files and csv files. As the script runs an output will appear for each file letting the user know if a file got skipped due to compatibility issues with SPINDLE or if the script successfully ran it. 

### spindle_single_pipeline_2.py3 flags: 

-h (help shows the flags and exits)

-f (files which can be a list of independent starfiles separated by a space or is the location of a tgz file)*

-t (used to target specific files in the tgz file)

-models (location of SPINDLE models directory)*

-out (the name of a new directory being made that will contain all the pdf and csv files being created)*

*Needed for script to run

Note: If one runs a full tgz file without using the -t flag the script will run everything. It is unnecessary to use the -t flag if just running independent starfiles. Additionally, this script needs the spindle_calibrations.py to be in the same directory to run, as this script contains the calibrations needed for the models to run correctly. Finally, the script needs to be run in the same environment as SPINDLE (virtual environment, etc.) so that the models of SPINDLE load. 

Example of running only independent files: ./spindle_single_pipeline_2.py3 -f 50285 50284 50283 -models ../../models -out Independent_BMRB_models

Example of running a tgz file and extracting specific files: ./spindle_single_pipeline_2.py3 -f  -models ../../models -out Independent_BMRB_models

### Output of Script

The output for each starfile is an independent pdf file for each condition/field for S<sub>2</sub> and R<sub>ex</sub>. This pdf file will contain either the fitting of S<sub>2</sub> or R<sub>ex</sub> for that condition/field of SPINDLE vs the BMRB file data by residue along with the correlating global tumbling time. The csv file will contain residue number; S<sub>2</sub> and TauE reported from the BMRB; S<sub>2</sub>, S<sub>2</sub> error, TauE, TauE error, R<sub>ex</sub>, and Rex error predicted from SPINDLE; the quality of fitting; TauC reported by SPINDLE; the field; and the condition for all of the conditions/fields in that one file.

## To Look at Global Tumbling Time

To compare the reported global tumbling time from the BMRB starfiles and SPINDLE one must use the compare_TauC_4.py3 script. Note, the global tumbling time nor if the model is isotropic, axial, or anisotropic are values that are not reported anywhere in the starfile or related BMRB entry page, so each time and fit of model was manually reported. The reports were put in an excel file and this excel file was run. The script then reads in an excel file and the resulting summary csv file from spindle_pipline_9.py3. The output is a directory containing a pdf file, text file, and csv file. 

### Compare_TauC_4.py3 flags:

-h (help flag to print reminders of each flag)

-spindle (path to SPINDLE summary csv)*

-exp (path to excel or csv file with TauC’s)*

-out (name of created directory to save outputs)*

-sheet (name of specific excel sheet one would like to load, default name chosen is “Curated”)

*Needed for script to run

Note: The -sheet flag is optional and will default to looking for a sheet with “Curated” if there are multiple sheets in the file. The script can be run completely out of the SPINDLE environment as it does not rely on SPINDLE’s models for any analysis.

Example of running: ./compare_TauC_4.py3 -spindle ../../models -exp Just_TauC_List.xlsx -out Final_TauC

### Output of Script
The output includes a pdf of a figure comparing the model predicted and reported TauC. The figure also will separate the isotropic fitted models compared to other fitted models through different colors along with the reported statics reflecting how well everything correlates. Additionally, a text file with the reported statics is generated along with a csv file including the extracted information from the excel file and from the model. 

