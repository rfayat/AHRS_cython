cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Cprodqv(np.ndarray[DTYPE_t, ndim=1] q,
             np.ndarray[DTYPE_t, ndim=1] v):
   "Rotate a vector v (ndarray, (3,)) by a quaternion q (ndarray, (4,))"
   cdef double r1,r2,r3 # elements of the rotated vector
   cdef double q0,q1,q2,q3,q0q0,q0q1,q0q2,q0q3,q1q1,q1q2,q1q3,q2q2,q2q3,q3q3

   # grab the quaternion's value
   q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

   # Auxiliary variables to avoid repeated arithmetic
   q0q0, q0q1, q0q2, q0q3 = 2*q0*q0, 2*q0*q1, 2*q0*q2, 2*q0*q3
   q1q1, q1q2, q1q3       =          2*q1*q1, 2*q1*q2, 2*q1*q3
   q2q2, q2q3             =                   2*q2*q2, 2*q2*q3
   q3q3                   =                            2*q3*q3

   # rotate the vector
   r1 = (q0q0-1+q1q1) * v[0] +  (q1q2+q0q3)  * v[1] +  (q1q3-q0q2) * v[2]
   r2 =  (q1q2-q0q3)  * v[0] + (q0q0-1+q2q2) * v[1] +  (q2q3+q0q1) * v[2]
   r3 =  (q1q3+q0q2)  * v[0] +  (q2q3-q0q1)  * v[1] + (q0q0-1+q3q3)* v[2]

   return(r1, r2, r3)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Cprodqq(np.ndarray[DTYPE_t, ndim=1] q,
             np.ndarray[DTYPE_t, ndim=1] r):
    "Quaternion product q * r where q and r are two np arrays with 4 elements."
    cdef double s0, s1, s2, s3 # elements of the output quaternion
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    r0, r1, r2, r3 = r[0], r[1], r[2], r[3]

    s0 = q0*r0 - q1*r1 - q2*r2 - q3*r3
    s1 = q0*r1 + q1*r0 + q2*r3 - q3*r2
    s2 = q0*r2 - q1*r3 + q2*r0 + q3*r1
    s3 = q0*r3 + q1*r2 - q2*r1 + q3*r0

    return(s0, s1, s2, s3)

@cython.boundscheck(False)
@cython.wraparound(False)
def CprodQv(np.ndarray[DTYPE_t, ndim=2] Q,
            np.ndarray[DTYPE_t, ndim=1] v):
  """
  Rotate a 3D vector by an array of quaternions.

  INPUT
  *v, ndarray (3,): vector to rotate
  *Q, ndarray (nQuaternion, 4): quaternions to use to rotate v.

  RETURN
  *rotatedVectors (nQuaternion, 3): rotated vectors
  """
  cdef int nQuaternion = Q.shape[0] # total number of quaternions used
  cdef np.ndarray[DTYPE_t, ndim=2] rotatedVectors = np.zeros((nQuaternion, 3), dtype=np.float64) # array of rotated vectors
  cdef int quaternionIndex

  for quaternionIndex in range(nQuaternion):
    quaternion = Q[quaternionIndex]
    rotatedVectors[quaternionIndex] = Cprodqv(quaternion, v)

  return(rotatedVectors)
