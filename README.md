
Jupyter Notebook code to compute the K-operator (Mannone, Fazio, Marwan, 2024), from fMRI data of selected Parkinson's disease patients and healthy controls from the PPMI dataset (Mannone, Fazio, Ribino, Marwan, 2024). Because of the length and size of the complete code, only a part of it is shown here. All the exploited functions and features are present. By selecting different patients, the visualizations and computations are updated.
Libraries nilearn and dicom2nifti have been used, an in particular, the visualization exploited by S. Hough et al. The computation of the K-operator is original (by M. Mannone).

The articles can be respectively retrieved at:
https://pubs.aip.org/aip/cha/article/34/5/053133/3294604/Modeling-a-neurological-disorder-as-the-result-of
https://link.springer.com/article/10.1140/epjs/s11734-024-01345-6

The file neuronal_simulation_maria_2024.py is a small addition, created with the help of chatGPT, to simulate a neural population of neurons with the Leaky Integrate-and-Fire (LIF). The synaptic weights density values are empirically chosen, according to the retrieved values deriving from the analysis of time series in real data. The results obtained with this code constitute a small part of a recent submission (Mannone, Ribino, Marwan, Fazio, 2024).

---

In April 2026, I correct a small part of the code, that does not affect the core results.
In fact, the tensor product we are using is conceptual (action on different spaces).
To correct figure 2, the line
np.kron(submatrix, matrix2 @ inverse_matrix1)
is now replaced with

K = matrix2 @ inverse_matrix1
def create_tensor_product_submatrix(n, i, j):
    submatrix = np.zeros((n, n))
    submatrix[i, j] = K_approximation[i, j]
    return submatrix

And we get the visualization with:
tensor_product_all_submatrices = np.zeros_like(K)
for i in range(n):
    for j in range(n):
        tensor_product_all_submatrices += create_tensor_product_submatrix(n, i, j)
 result_matrix = tensor_product_all_submatrices
