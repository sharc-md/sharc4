#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************


import numpy as np

from error import Error


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Estimate a rotation to optimally align two sets of vectors.
    Find a rotation between frames A and B which best aligns a set of
    vectors `a` and `b` observed in these frames. The following loss
    function is minimized to solve for the rotation matrix
    :math:`C`:
    .. math::
        L(C) = \\frac{1}{2} \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{a}_i -
        C \\mathbf{b}_i \\rVert^2 ,
    where :math:`w_i`'s are the `weights` corresponding to each vector.
    The rotation is estimated with Kabsch algorithm.
    Parameters
    ----------
    a : array_like, shape (N, 3)
        Vector components observed in reference frame A. Each row of `a`
        denotes a vector.
    b : array_like, shape (N, 3)
        Vector components observed in another frame B. Each row of `b`
        denotes a vector.

    Returns
    ----------
    B : rotation matrix in cartesian coordinates that rotates `a` onto `b`
    a_s : center of mass of `a`
    b_s : center of mass of `b`

    B, a_s, b_s = kabsch(a, b)
    a_in_frame_of_b = (a - a_s) @ B + b_s
    b_in_frame_of_a = (b - b_s) @ B.T + a_s
    """
    if a.ndim != 2 or a.shape[-1] != 3:
        raise ValueError("Expected input `a` to have shape (N, 3), " "got {}".format(a.shape))
    if b.ndim != 2 or b.shape[-1] != 3:
        raise ValueError("Expected input `b` to have shape (N, 3), " "got {}.".format(b.shape))

    if a.shape != b.shape:
        raise ValueError(
            "Expected inputs `a` and `b` to have same shapes" ", got {} and {} respectively.".format(a.shape, b.shape)
        )
    # shift to centroid
    a_s = sum(a) / a.shape[0]
    b_s = sum(b) / b.shape[0]

    # shift b to a
    B = np.einsum("ji,jk->ik", a - a_s, b - b_s)
    # B = a.T @ (b + (a_s - b_s))
    u, s, vT = np.linalg.svd(B)

    B = u @ vT
    if np.linalg.det(B) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
        B = u @ vT  # alternatively: B -= np.einsum('i,k->ik', 2*u[:,-1], vT[-1,:])

    if s[1] + s[2] < 1e-16 * s[0]:
        print("Optimal rotation is not uniquely or poorly defined for the given sets of vectors.")

    return B, a_s, b_s


def kabsch_w(a: np.ndarray, b: np.ndarray, weights) -> np.ndarray:
    """Estimate a rotation to optimally align two sets of vectors.
    Find a rotation between frames A and B which best aligns a set of
    vectors `a` and `b` observed in these frames. The following loss
    function is minimized to solve for the rotation matrix
    :math:`C`:
    .. math::
        L(C) = \\frac{1}{2} \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{a}_i -
        C \\mathbf{b}_i \\rVert^2 ,
    where :math:`w_i`'s are the `weights` corresponding to each vector.
    The rotation is estimated with Kabsch algorithm.
    Parameters
    ----------
    a : array_like, shape (N, 3)
        Vector components observed in reference frame A. Each row of `a`
        denotes a vector.
    b : array_like, shape (N, 3)
        Vector components observed in another frame B. Each row of `b`
        denotes a vector.
    weights : array_like shape (N,), optional
        Weights describing the relative importance of the vector
        observations.

    Returns
    ----------
    B : rotation matrix in cartesian coordinates that rotates `a` onto `b`
    a_s : center of mass of `a` with weights
    b_s : center of mass of `b` with weights

    B, a_s, b_s = kabsch(a, b, weights)
    a_in_frame_of_b = (a - a_s) @ B + b_s
    b_in_frame_of_a = (b - b_s) @ B.T + a_s
    """
    if a.ndim != 2 or a.shape[-1] != 3:
        raise Error("Expected input `a` to have shape (N, 3), " "got {}".format(a.shape))
    if b.ndim != 2 or b.shape[-1] != 3:
        raise Error("Expected input `b` to have shape (N, 3), " "got {}.".format(b.shape))

    if a.shape != b.shape:
        raise Error("Expected inputs `a` and `b` to have same shapes" ", got {} and {} respectively.".format(a.shape, b.shape))

    weights = np.asarray(weights)
    if weights.ndim != 1 or weights.shape[0] != a.shape[0]:
        raise ValueError(f"Expected input `weights` to have shape {a.shape[0]}")
    # shift to centroid
    M = sum(weights)
    a_s = weights @ a / M
    b_s = weights @ b / M
    B = np.einsum("j,ji->ji", weights, a - a_s)
    B = np.einsum("ji,jk->ik", B, b - b_s)
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


def kabsch_w_with_deriv(a: np.ndarray, b: np.ndarray, weights) -> (np.ndarray, float, float, np.ndarray):
    """Estimate a rotation to optimally align two sets of vectors.
    Find a rotation between frames A and B which best aligns a set of
    vectors `a` and `b` observed in these frames. The following loss
    function is minimized to solve for the rotation matrix
    :math:`C`:
    .. math::
        L(C) = \\frac{1}{2} \\sum_{i = 1}^{n} w_i \\lVert \\mathbf{a}_i -
        C \\mathbf{b}_i \\rVert^2 ,
    where :math:`w_i`'s are the `weights` corresponding to each vector.
    The rotation is estimated with Kabsch algorithm.
    Parameters
    ----------
    a : array_like, shape (N, 3)
        Vector components observed in reference frame A. Each row of `a`
        denotes a vector.
    b : array_like, shape (N, 3)
        Vector components observed in another frame B. Each row of `b`
        denotes a vector.
    weights : array_like shape (N,), optional
        Weights describing the relative importance of the vector
        observations.

    Returns
    ----------
    B : rotation matrix in cartesian coordinates that rotates `a` onto `b`
    a_s : center of mass of `a` with weights
    b_s : center of mass of `b` with weights

    B, a_s, b_s = kabsch(a, b, weights)
    a_in_frame_of_b = (a - a_s) @ B + b_s
    b_in_frame_of_a = (b - b_s) @ B.T + a_s
    """
    if a.ndim != 2 or a.shape[-1] != 3:
        raise Error("Expected input `a` to have shape (N, 3), " "got {}".format(a.shape))
    if b.ndim != 2 or b.shape[-1] != 3:
        raise Error("Expected input `b` to have shape (N, 3), " "got {}.".format(b.shape))

    if a.shape != b.shape:
        raise Error("Expected inputs `a` and `b` to have same shapes" ", got {} and {} respectively.".format(a.shape, b.shape))

    weights = np.asarray(weights)
    if weights.ndim != 1 or weights.shape[0] != a.shape[0]:
        raise ValueError(f"Expected input `weights` to have shape {a.shape[0]}")
    # shift to centroid
    M = sum(weights)
    a_s = weights @ a / M
    b_s = weights @ b / M
    A = np.einsum("j,ji->ji", weights, a - a_s)
    A = np.einsum("ji,jk->ik", A, b - b_s)
    U, S, V_T = np.linalg.svd(A)

    det_U_Vt = np.linalg.det(U @ V_T)
    L = np.eye(3)
    L[2, 2] = det_U_Vt
    B = U @ L @ V_T

    if S[1] + S[2] < 1e-16 * S[0]:
        print("Optimal rotation is not uniquely or poorly defined " "for the given sets of vectors.")
        raise ValueError("Optimal rotation is not uniquely or poorly defined ")

    na = a.shape[0]
    dr = np.eye(na * 3)
    dr = dr.reshape((na * 3, na, 3)) * weights[None, :, None]
    dr = dr.reshape((na, 3, na * 3))
    dA_full = np.einsum("ax,ayk->kxy", a - a_s, dr)

    F = np.array([[1 / (s2**2 - s1**2) if i != j else 0.0 for j, s2 in enumerate(S)] for i, s1 in enumerate(S)], dtype=float)
    S_inv = np.diag([1 / s if s > 1e-15 else 0 for s in S])
    S = np.diag(S)
    eye_3 = np.eye((3), dtype=float)
    V = V_T.T
    dB = np.zeros((na * 3, 3, 3), dtype=float)

    for i in range(na * 3):
        dA = dA_full[i, ...]
        dU = U @ (F * (U.T @ dA @ V @ S + S @ V_T @ dA.T @ U)) + (eye_3 - U @ U.T) @ dA @ V @ S_inv
        dV = V @ (F * (S @ U.T @ dA @ V + V_T @ dA.T @ U @ S)) + (eye_3 - V @ V_T) @ dA.T @ U @ S_inv
        dB[i, ...] = dU @ L @ V.T + U @ L @ dV.T
    return B, a_s, b_s, dB
