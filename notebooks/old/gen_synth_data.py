import numpy as np
import h5py
#import datetime
#datetime.datetime.now().strftime('%Y%m%d%H%M%S')

#Configure these params!
N = 250
dt = 0.004
tau = 0.010
gamma = 2.5
T_hours = 24.0
T = T_hours*60**2

#Get file
f = h5py.File("rnn_data.hdf5", "a")
f.attrs["dt"] = dt
f.attrs["tau"] = tau
f.attrs["gamma"] = gamma
f.attrs["N"] = N

#Delete old data it exists
if "data" in f.keys():
    del f["data"]

#Weight variance ~1/N
W = np.random.normal(0, 1/np.sqrt(N), (N, N))

#Define differential equation
def dx_dt(x, t):
    x_dot = (1/tau)*(-x + gamma*np.dot(W, np.tanh(x)))
    return x_dot

#Random IC
x_0 = np.random.normal(0, 1, N)

#Create dataset
num_timesteps = int(np.round(T / dt))
X = f.create_dataset("data", (num_timesteps, N), dtype=np.float64)

#Generate RNN data
X[0, :] = x_0
for i in range(1, num_timesteps):
    if i % 2500 == 0:
        print(str(np.round((i / num_timesteps)*100, 2))+"%")
    X[i, :] = X[i-1, :] + dt*dx_dt(X[i-1, :], i*dt)