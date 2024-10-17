
Jupyter Notebook code to compute the K-operator (Mannone, Fazio, Marwan, 2024), from fMRI data of selected Parkinson's disease patients and healthy controls from the PPMI dataset (Mannone, Fazio, Ribino, Marwan, 2024). Because of the length and size of the complete code, only a part of it is shown here. All the exploited functions and features are present. By selecting different patients, the visualizations and computations are updated.
Libraries nilearn and dicom2nifti have been used, an in particular, the visualization exploited by S. Hough et al. The computation of the K-operator is original (by M. Mannone).

The articles can be respectively retrieved at:
https://pubs.aip.org/aip/cha/article/34/5/053133/3294604/Modeling-a-neurological-disorder-as-the-result-of
https://link.springer.com/article/10.1140/epjs/s11734-024-01345-6

The file neuronal_simulation_maria_2024.py is a small addition, created with the help of chatGPT, to simulate a neural population of neurons with the Leaky Integrate-and-Fire (LIF). The synaptic weights density values are empirically chosen, according to the retrieved values deriving from the analysis of time series in real data. The results obtained with this code constitute a small part of a recent submission (Mannone, Ribino, Marwan, Fazio, 2024).

