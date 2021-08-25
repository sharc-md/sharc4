#!/usr/bin/env python3
import numpy as np

from error import Error

def kabsch(a, b) -> np.ndarray:
    """Estimate a rotation to optimally align two sets of vectors.
    Find a rotation between frames A and B which best aligns a set of
    vectors `a` and `b` observed in these frames. The following loss
    function is minimized to solve for the rotation matrix
    :math:`C`:
    .. math::
        L(C) = \\frac{1}{2} \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{a}_i -
        C \\mathbf{b}_i \\rVert^2 ,
    where :math:`w_i`'s are the `weights` corresponding to each vector.
    The rotation is estimated with Kabsch algorithm [1]_.
    Parameters
    ----------
    a : array_like, shape (N, 3)
        Vector components observed in initial frame A. Each row of `a`
        denotes a vector.
    b : array_like, shape (N, 3)
        Vector components observed in another frame B. Each row of `b`
        denotes a vector.
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[-1] != 3:
        raise ValueError("Expected input `a` to have shape (N, 3), " "got {}".format(a.shape))
    b = np.asarray(b)
    if b.ndim != 2 or b.shape[-1] != 3:
        raise ValueError("Expected input `b` to have shape (N, 3), " "got {}.".format(b.shape))

    if a.shape != b.shape:
        raise ValueError(
            "Expected inputs `a` and `b` to have same shapes"
            ", got {} and {} respectively.".format(a.shape, b.shape)
        )
    # shift to centroid
    a_s = sum(a) / a.shape[0]
    b_s = sum(b) / b.shape[0]

    #shift b to a
    B = np.einsum('ji,jk->ik', a - a_s, b - b_s)
    # B = a.T @ (b + (a_s - b_s))
    u, s, vT = np.linalg.svd(B)

    B = u @ vT
    if np.linalg.det(B) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
        B = u @ vT    # alternatively: B -= np.einsum('i,k->ik', 2*u[:,-1], vT[-1,:])

    if s[1] + s[2] < 1e-16 * s[0]:
        print("Optimal rotation is not uniquely or poorly defined " "for the given sets of vectors.")

    return B, a_s, b_s


def kabsch_w(a, b, weights) -> np.ndarray:
    """Estimate a rotation to optimally align two sets of vectors.
    Find a rotation between frames A and B which best aligns a set of
    vectors `a` and `b` observed in these frames. The following loss
    function is minimized to solve for the rotation matrix
    :math:`C`:
    .. math::
        L(C) = \\frac{1}{2} \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{a}_i -
        C \\mathbf{b}_i \\rVert^2 ,
    where :math:`w_i`'s are the `weights` corresponding to each vector.
    The rotation is estimated with Kabsch algorithm [1]_.
    Parameters
    ----------
    a : array_like, shape (N, 3)
        Vector components observed in initial frame A. Each row of `a`
        denotes a vector.
    b : array_like, shape (N, 3)
        Vector components observed in another frame B. Each row of `b`
        denotes a vector.
    weights : array_like shape (N,), optional
        Weights describing the relative importance of the vector
        observations."""
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[-1] != 3:
        raise Error("Expected input `a` to have shape (N, 3), " "got {}".format(a.shape))
    b = np.asarray(b)
    if b.ndim != 2 or b.shape[-1] != 3:
        raise Error("Expected input `b` to have shape (N, 3), " "got {}.".format(b.shape))

    if a.shape != b.shape:
        raise Error(
            "Expected inputs `a` and `b` to have same shapes"
            ", got {} and {} respectively.".format(a.shape, b.shape)
        )

    weights = np.asarray(weights)
    if weights.ndim != 1 or weights.shape[0] != a.shape[0]:
        raise ValueError(f"Expected input `weights` to have shape {a.shape[0]}")
    # shift to centroid
    M = sum(weights)
    a_s = weights @ a / M
    b_s = weights @ b / M
    B = np.einsum('j,ji->ji', weights, a - a_s)
    B = np.einsum('ji,jk->ik', B, b - b_s)
    u, s, vT = np.linalg.svd(B)

    B = u @ vT
    # Correct improper rotation if necessary (as in Kabsch algorithm)
    if np.linalg.det(u @ vT) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
        B = u @ vT

    if s[1] + s[2] < 1e-16 * s[0]:
        print("Optimal rotation is not uniquely or poorly defined " "for the given sets of vectors.")

    return B, a_s, b_s
