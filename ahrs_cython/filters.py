"""Python wrapper for the cython implementation of AHRS filters.

Author: Romain Fayat, August 2024
"""
import numpy as np
from ahrs.filters import Madgwick as Madgwick_ahrs
from ahrs.filters import EKF as EKF_ahrs
from ahrs.common.orientation import acc2q, am2q
from .common import CprodQv
from .madgwick import madgwickIMU, madgwickAHRS
from .ekf import ekfIMU


def _input_sanity_check(gyr, acc, mag=None):
    "Sanity check on the input data (matching shapes, no missing values)."
    # Make sure the shapes of the input data are consistent
    if acc.shape != gyr.shape:
        raise ValueError("acc and gyr are not the same size")
    if mag is not None and mag.shape != gyr.shape:
        raise ValueError("mag and gyr are not the same size")
    # Check for nan values in the input data
    if np.any(np.isnan(acc)):
        raise ValueError("nan accelerometer values are not supported")
    if np.any(np.isnan(gyr)):
        raise ValueError("nan gyroscope values are not supported")
    if mag is not None and np.any(np.isnan(mag)):
        raise ValueError("nan gyroscope values are not supported")


class Madgwick(Madgwick_ahrs):
    """Wrapper for Cython implementation of the Madgwick filter.

    The API and documentation are directly taken from the ahrs module.
    """

    __doc__ += "\n" + Madgwick_ahrs.__doc__

    def _compute_all(self):
        "Compute all quaternions from the input data."
        self._compute_q0()  # initial quaternion
        _input_sanity_check(self.gyr, self.acc, self.mag)
        # IMU pipeline if no mag data was provided
        if self.mag is None:
            return madgwickIMU(self.acc, self.gyr, self.q0,
                               self.gain, self.frequency)[:-1]
        # MARG pipeline if mag data was provided
        else:
            return madgwickAHRS(self.acc, self.gyr, self.mag, self.q0,
                                self.gain, self.frequency)[:-1]

    def _compute_q0(self):
        "Compute the initial quaternion for the iterative algorithm."
        if self.q0 is not None:
            self.q0 /= np.linalg.norm(self.q0)
        elif self.mag is None:
            self.q0 = acc2q(self.acc[0])
        else:
            self.q0 = am2q(self.acc[0], self.mag[0])

    def gravity_estimate(self):
        "Estimate the coordinates of gravity in the sensor reference frame."
        if not hasattr(self, "Q"):
            raise ValueError(
                "The object was not instantiated with at least accelerometer and gyroscope data."  # noqa E501
            )
        return CprodQv(self.Q, np.array([0, 0, 1], dtype=float))


class EKF(EKF_ahrs):
    """Wrapper for Cython implementation of the Extended Kalman filter.

    The API and documentation are directly taken from the ahrs module.
    """

    __doc__ += "\n" + EKF_ahrs.__doc__

    def _compute_all(self, frame):
        "Compute all quaternions from the input data."
        self._compute_q0()  # initial quaternion
        _input_sanity_check(self.gyr, self.acc, self.mag)
        if self.mag is not None:
            raise(NotImplementedError, "EKF with input mag data is not implemented.")
        return ekfIMU(self.acc, self.gyr, self.q0, self.frequency,
                      self.g_noise, self.a_ref, self.R, self.P)[:-1]
    
    def _compute_q0(self):
        "Compute the initial quaternion for the iterative algorithm."
        if self.q0 is not None:
            self.q0 /= np.linalg.norm(self.q0)
        elif self.mag is None:
            self.q0 = acc2q(self.acc[0])
        else:
            self.q0 = am2q(self.acc[0], self.mag[0])
    
    def gravity_estimate(self):
        "Estimate the coordinates of gravity in the sensor reference frame."
        if not hasattr(self, "Q"):
            raise ValueError(
                "The object was not instantiated with at least accelerometer and gyroscope data."  # noqa E501
            )
        return CprodQv(self.Q, np.array([0, 0, 1], dtype=float))
