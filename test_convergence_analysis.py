import numpy as np
from convergence_analysis import ConvergenceAnalyzer as ca

def test_matrix_convergence():
    A = [
        [
            [2,2],
            [2,2]
        ],
        [
            [1,1],
            [1,1]
        ]
    ]
    A = np.array(A)

    optimal = [
        [1,1],
        [1,1]
    ]
    optimal = np.array(optimal)

    expected_norm = np.array((1,0))

    np.testing.assert_allclose(expected_norm, 
    ca.calc_mse_along_rows(A, optimal))

    