cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def q2r(np.ndarray[DTYPE_t, ndim=1] q):
    cdef double q0, q1, q2, q3
    cdef np.ndarray[DTYPE_t, ndim=1] q_normalized = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q_normalized
    cdef np.ndarray[DTYPE_t, ndim=2] rotmat = np.array([
        [1.0-2.0*(q2**2+q3**2), 2.0*(q1*q2-q0*q3), 2.0*(q1*q3+q0*q2)],
        [2.0*(q1*q2+q0*q3), 1.0-2.0*(q1**2+q3**2), 2.0*(q2*q3-q0*q1)],
        [2.0*(q1*q3-q0*q2), 2.0*(q0*q1+q2*q3), 1.0-2.0*(q1**2+q2**2)]
    ]) 
    return rotmat


@cython.boundscheck(False)
@cython.wraparound(False)
cdef skew(np.ndarray[DTYPE_t, ndim=1] x):
    cdef np.ndarray[DTYPE_t, ndim=2] y
    y = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0.0]
    ])
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
cdef compute_omega_matrix(np.ndarray[DTYPE_t, ndim=1] x):
    cdef np.ndarray[DTYPE_t, ndim=2] omega_matrix = np.array(
        [[0.0,  -x[0], -x[1], -x[2]],
         [x[0],   0.0,  x[2], -x[1]],
         [x[1], -x[2],   0.0,  x[0]],
         [x[2],  x[1], -x[0],   0.0]],
        dtype=np.float64)

    return omega_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
cdef dfdq(np.ndarray[DTYPE_t, ndim=1] omega, double dt):
    cdef np.ndarray[DTYPE_t, ndim=1] x = .5 * dt * omega
    return compute_omega_matrix(x) + np.identity(4, dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef dhdq(np.ndarray[DTYPE_t, ndim=1] q, np.ndarray[DTYPE_t, ndim=1] a_ref):
    cdef double qw, qx, qy, qz
    qw, qx, qy, qz = q

    cdef np.ndarray[DTYPE_t, ndim=1] v = a_ref
    cdef np.ndarray[DTYPE_t, ndim=2] H = np.array([
        [-qy*v[2] + qz*v[1],  qy*v[1] + qz*v[2], -qw*v[2] + qx*v[1] - 2.0*qy*v[0],  qw*v[1] + qx*v[2] - 2.0*qz*v[0]],
        [ qx*v[2] - qz*v[0],  qw*v[2] - 2.0*qx*v[1] + qy*v[0],  qx*v[0] + qz*v[2], -qw*v[0] + qy*v[2] - 2.0*qz*v[1]],
        [-qx*v[1] + qy*v[0], -qw*v[1] - 2.0*qx*v[2] + qz*v[0],  qw*v[0] - 2.0*qy*v[2] + qz*v[1],  qx*v[0] + qy*v[1]]
    ])
    # HERE DROPPED CASE WHERE len(z) > 6
    return 2.0 * H


@cython.boundscheck(False)
@cython.wraparound(False)
cdef f(np.ndarray[DTYPE_t, ndim=1] q, np.ndarray[DTYPE_t, ndim=1] omega, double dt):
    cdef np.ndarray[DTYPE_t, ndim=2] omega_t
    cdef np.ndarray[DTYPE_t, ndim=1] q_t

    omega_t = compute_omega_matrix(omega)
    q_t = np.matmul(np.identity(4, dtype=np.float64) + 0.5 * dt * omega_t, q)
    return q_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef h(np.ndarray[DTYPE_t, ndim=1] q, np.ndarray[DTYPE_t, ndim=1] a_ref):
    cdef np.ndarray[DTYPE_t, ndim=2] rotmat = q2r(q)
    cdef np.ndarray[DTYPE_t, ndim=2] C = np.transpose(rotmat)

    return np.matmul(C, a_ref)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef ekfIMUStep(np.ndarray[DTYPE_t, ndim=1] accStep,
                np.ndarray[DTYPE_t, ndim=1] gyrStep,
                np.ndarray[DTYPE_t, ndim=1] quatStep,
                double dt,
                double g_noise,
                np.ndarray[DTYPE_t, ndim=1] a_ref,
                np.ndarray[DTYPE_t, ndim=2] R,
                np.ndarray[DTYPE_t, ndim=2] P):
    cdef np.ndarray[DTYPE_t, ndim=1] z, q_t, v, q, y
    cdef np.ndarray[DTYPE_t, ndim=2] H, S, Q_t, W, P_t, F, K, S_inv, P_new
    cdef double a_norm = np.linalg.norm(accStep)
    if a_norm < 1e-6:
        return quatStep
    z = accStep / a_norm
    q_t = f(quatStep, gyrStep, dt)
    F = dfdq(gyrStep, dt)
    # W   = 0.5*self.Dt * np.r_[[-q[1:]], q[0]*np.identity(3) + skew(q[1:])]
    W = .5 * dt * np.vstack((-quatStep[1:].reshape(1, -1), quatStep[0] * np.identity(3, dtype=np.float64) + skew(quatStep[1:])))
    Q_t = .5 * dt * g_noise * np.matmul(W, np.transpose(W))
    P_t = np.matmul(np.matmul(F, P), np.transpose(F)) + Q_t
    
    y = h(q_t, a_ref)
    v = z - y
    H = dhdq(q_t, a_ref)
    S = np.matmul(np.matmul(H, P_t), np.transpose(H)) + R
    S_inv = np.linalg.inv(S)
    K = np.matmul(np.matmul(P_t, np.transpose(H)), S_inv)
    P_new = np.matmul(np.identity(4) - np.matmul(K, H), P_t)
    q = q_t + np.matmul(K, v)
    q = q / np.linalg.norm(q)
    return q, P_new


@cython.boundscheck(False)
@cython.wraparound(False)
def ekfIMU(np.ndarray[DTYPE_t, ndim=2] acc,
           np.ndarray[DTYPE_t, ndim=2] gyr,
           np.ndarray[DTYPE_t, ndim=1] quat0,
           double sampleFreq,
           double g_noise,
           np.ndarray[DTYPE_t, ndim=1] a_ref,
           np.ndarray[DTYPE_t, ndim=2] R,
           np.ndarray[DTYPE_t, ndim=2] P0):
    cdef np.ndarray[DTYPE_t, ndim=2] P
    cdef double dt = 1 / sampleFreq
    cdef int n_row = acc.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] quat = np.zeros((n_row + 1, 4), dtype=np.float64)
    quat[0,:] = quat0
    P = P0

    for row in range(n_row):
        quat[row+1], P_new = ekfIMUStep(acc[row], gyr[row], quat[row],
                                        dt, g_noise, a_ref, R, P)
        P = P_new

    return(quat)
