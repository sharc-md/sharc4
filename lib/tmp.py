import numpy as np

def concatenate_arrays(arr1, arr2):
    """
    Concatenate each row of arr1 with each row of arr2 and return a 3D array.

    Parameters:
    - arr1: numpy array, shape (M, N)
    - arr2: numpy array, shape (M, N)

    Returns:
    - result: numpy array, shape (M, M, 2N)
    """
    M, N = arr1.shape

    # Reshape the arrays to ensure they have the same number of columns (2N)
    arr1_reshaped = arr1.reshape((M, 1, N))
    arr2_reshaped = arr2.reshape((1, M, N))

    # Concatenate along the last axis to form the 3D array
    result = np.concatenate((arr1_reshaped, arr2_reshaped), axis=2)

    return result

# Example usage:
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

result = concatenate_arrays(arr1, arr2)
print(result)

