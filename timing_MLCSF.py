# calculates the timing for MLCSF runs

import json
import numpy as np

# constants to use later
EXP1, EXP2 = range(2)
exp_conditions = ["active", "aprior", "random", "rprior"]
phenotypes_exp1 = [ 'Normal', 'Mild Amblyopia', 'Cataracts', 'Multiple Sclerosis']
phenotypes_exp2 = ["Results"]

# files used for manuscript
experimentToResults = {
    EXP1: "saved_results/Figure06/" +
    "results_2023-08-25_22-04-57.json",
    EXP2: "saved_results/Figure09/" +
    "results_2023-08-30_19-46-46.json",
}

for exp, filePath in experimentToResults.items():

    with open(filePath, 'r') as file:
        results = json.load(file)

    if exp == EXP1:
        phenotypes = phenotypes_exp1
        numExps = len(results["Normal"])
        print("Experiment 1")
    else:
        phenotypes = phenotypes_exp2
        numExps = len(results[phenotypes[0]])
        print("Experiment 2")
        
    timepointsToIndex = {
        10: 0,
        20: 1,
        50: 2,
        100: 3
    }
    
    allExperimentTimes = np.zeros((len(exp_conditions), numExps, len(timepointsToIndex)))
    
    # either average over Normal, MS, etc or average across all SZ and NT
    for phenotype in phenotypes:
        print(phenotype if exp == EXP1 else "SZ and NT")
        print("Number of Points \t Time Elapsed")

        if phenotype == "Multiple Sclerosis": phenotype = "MS"
        
        if exp == EXP2: print(len(results["Results"]))

        # store all timepoints to average later
        for j, exp_results in enumerate(results[phenotype]):
            for i, condition in enumerate(exp_conditions):
                for timepoint, time in exp_results[f"{condition}_times"]:
                    k = timepointsToIndex[timepoint]
                    allExperimentTimes[i, j, k] = time
        
        means = allExperimentTimes[i,:,:].mean(axis=0)
        stds = allExperimentTimes[i,:,:].std(axis=0)
    
        for i, timepoint in enumerate(timepointsToIndex):
            print(f"{timepoint} \t\t\t {means[i]:.2f} ± {stds[i]:.2f}s")
    
        print()