"""
Run EKF algorithm on an example dataset.

Author: Romain Fayat

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ahrs_cython.filters import EKF
from cycler import cycler
from scipy.spatial.transform import Rotation as R


SR = 300. # Hz
COL_ACC = ["ax", "ay", "az"]
COL_GYR = ["gx", "gy", "gz"]
COL_ACC_G = ["ax_G", "ay_G", "az_G"]
VAR_ACC = 2e-3
VAR_GYR = .75

# Colors for plotting
RGB = ["#D55E00", "#009E73", "#0072B2"]
RGB_CYCLER = cycler(color=RGB)

# Load the example dataset
data_path = "example/IMU_data_sample.csv"
df = pd.read_csv(data_path, index_col=0)
df["time"] = np.arange(len(df)) / SR


# Compute gravitational acceleration and add it to the dataframe
ekf_filter = EKF(gyr=np.radians(df[COL_GYR].values),
                 acc=df[COL_ACC].values,
                 frequency=SR,
                 var_acc=VAR_ACC,
                 var_gyr=VAR_GYR,
                 frame="ENU",
                 )

acc_G = pd.DataFrame(ekf_filter.gravity_estimate(),
                     index=df.index,
                     columns=COL_ACC_G)
df = df.join(acc_G)

# plot
t_start, t_end = 100, 120  # seconds
is_to_plot = (df.time >= t_start) & (df.time < t_end)
df_to_plot = df[is_to_plot]

fig, ax = plt.subplots()
ax.set_prop_cycle(RGB_CYCLER)
ax.plot(df_to_plot.time, df_to_plot[COL_ACC].values, alpha=.6, linewidth=.5)
ax.plot(df_to_plot.time, df_to_plot[COL_ACC_G].values)

ax.set_xlim([t_start, t_end])
ax.set_title("Example of gravity estimate using EKF")

fig.savefig("example/ekf_out.png")
