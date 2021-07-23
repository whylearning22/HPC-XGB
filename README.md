# A Unified Hierarchical XGBoost Model for Classifying Priorities for COVID-19 Vaccination Campaign

Python Code implementation for Luca Romeo, Emanuele Frontoni,
A Unified Hierarchical XGBoost Model for Classifying Priorities for COVID-19 Vaccination Campaign,
Pattern Recognition,2021,108197,ISSN 0031-3203,https://doi.org/10.1016/j.patcog.2021.108197.
The proposed HPC-XGB methodology learns to classify priority for COVID-19 vaccine administration using the Italian Federation of General Practitioners dataset, which is an Electronic Health Record data of 17k patients collected by 11 General Practitioners.


# Usage
*XGB_Vacc_Prior_ModelA.py* -->The first layer A focuses on the classification between the high vulnerable patients and the other categories. 

*XGB_Vacc_Prior_ModelB_binary.py*  --> The second layer  focuses on discriminating the priority classes in a specific age range categories (binary task).

*XGB_Vacc_Prior_ModelB_multi.py*  --> The second layer  focuses on discriminating the priority classes in a specific age range categories (multiclass task).

*select_pc.py*--> convert input/output data into the A, B task.

Notice how the validation procedure is not reported in the script, i.e. each split of the outer loop was trained with the optimal hyperparameters tuned in the inner loop (found in the previous experiments).

# Dataset
The Hierarchical Priority Classification eXtreme Gradient Boosting aims to provide priority classification for COVID-19 vaccine administration using the Italian Federation of General Practitioners dataset (FIMMG_COVID) that contains Electronic Health Record data of 17k patients. We measured the effectiveness of the proposed methodology for classifying all the priority classes while demonstrating a significant improvement with respect to the state of the art. The proposed ML approach, which is integrated into a clinical decision support system, is currently supporting General Pracitioners in assigning COVID-19 vaccine administration priorities to their assistants. https://mailchi.mp/netmedicaitalia/prisma-7577657
The FIMMG_COVID is available upon request at https://vrai.dii.univpm.it/content/fimmgcovid-dataset

# Acknowledgement
The present study has been supported by a research agreement among the Italian Federation of General Practitioner, Netmedica Italia and the Department of Information Engineering, Università Politecnica delle Marche, Ancona, Italy and by the "Microsoft Grant Award: AI for Health COVID-19". Authors would like to give a special thanks to Dr. Paolo Misericordia, Dr. Nicola Calabrese, Dr. Rino Moraglia, Ing. Alessandro Dalle Vedove and all 11 GPs (Core Data Team).

# Citation

If you find our code helpful in your resarch or work, please cite our paper.

@article{ROMEO2021108197,
title = {A Unified Hierarchical XGBoost Model for Classifying Priorities for COVID-19 Vaccination Campaign},
journal = {Pattern Recognition},
pages = {108197},
year = {2021},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2021.108197},
url = {https://www.sciencedirect.com/science/article/pii/S0031320321003794},
author = {Luca Romeo and Emanuele Frontoni},
}
