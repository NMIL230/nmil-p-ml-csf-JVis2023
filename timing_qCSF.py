# calculates the timing for qCSF runs

import json

from utility.utils import load_json_from_file
import numpy as np

EXP1, EXP2 = range(2)
phenotypes_exp1 = [ 'Normal', 'Mild Amblyopia', 'Cataracts', 'Multiple Sclerosis']

data_dir = 'data/raw_data/'
data_file_name = 'csf_curves_exp2.json'
data_file_path = f'{data_dir}{data_file_name}'

csf_curves_exp2 = load_json_from_file(data_file_path)

phenotypes_exp2 = [f"SZ{i}" for i in range(len(csf_curves_exp2['schizophrenia_participants']))]
phenotypes_exp2.extend([f"NT{i}" for i in range(len(csf_curves_exp2['neurotypical_participants']))])


######################
# SPECIFY FILES HERE #
######################
experimentToResults = {
    EXP1: "saved_results/Figure06/qcsf/" +
    "results_2023-08-29_19-02-26.json",
    EXP2: "saved_results/Figure08/qcsf/" +
    "results_2023-08-29_19-50-51.json",
}


for exp, filePath in experimentToResults.items():
    phenotypes = phenotypes_exp1 if exp == EXP1 else phenotypes_exp2

    with open(filePath, 'r') as file:
        results = json.load(file)
    
    timepoints = results["timepoints"]
    numExps = results["numExperiments"] if "numExperiments" in results else 1
    allExperimentTimes = np.zeros((len(phenotypes)*numExps, len(timepoints)))
    
    i = 0
    for phenotype in phenotypes:
        for exp_results in results[phenotype]:
            allExperimentTimes[i, :] = exp_results["times"] 
            i+=1
    
    print(f"Experiment {'1' if exp == EXP1 else '2'}")
    print("Number of Points \t Time Elapsed")
    means = allExperimentTimes.mean(axis=0)
    stds = allExperimentTimes.std(axis=0)
    
    for i, timepoint in enumerate(timepoints):
        print(f"{timepoint} \t\t\t {means[i]:.2f} ± {stds[i]:.2f}s")
    
    print()