"""
David Clark
March 2018

Referneces:
1. Welch, G. & Bishop, G. An Introduction to the Kalman Filter. In Practice 7, 1–16 (2006).
2. B. M. Yu, K. V. Shenoy, Derivation of Kalman Filtering and Smoothing Equations Forward Recursions : Filtering.
	Matrix, 1–5 (2004)
3. W. Q. Malik, W. Truccolo, E. N. Brown and L. R. Hochberg, "Efficient Decoding With Steady-State Kalman Filter in
	Neural Interface Systems," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 19, no. 1,
	pp. 25-34, Feb. 2011.
4. https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.linalg.solve_discrete_are.html
5. Gilja, V. et al. Technical Report : 1–49
"""

import numpy as np 
import scipy.linalg
import pykalman

"""
Run one time-step update for the Kalman filter

System matrix inputs:
A = dynamics matrix 
H = observation matrix 
Q = dynamics noise 
R = observation noise 

Current mean/covaraince inputs:
x_hat_km1 = current state mean
P_kma     = current state covariance 

Observation inputs:
z_k = new observation

Notation follows [1].

For a concise derivation, as well as derivation of the backwards update equations, see [2].

If refit_kf_inno_2 = True, then non-velocity-velocity components of the a priori state estiamte 
will be zeroed out (a "casual intervation") -- note that this assumes a 2D cursor scenario is being used
"""
def kf_update(A, H, Q, R, P_km1, x_hat_km1, z_k, refit_kf_inno_2=False):
	"""
	1: Time update ==> project state mean and covaraince one step in time
	('km1' means 'k minus 1')
	"""
	#Project state mean ahead using dynamics matirx
	x_hat_k_prior = np.dot(A, x_hat_km1)

	#Project state covaraince matrix ahead using dynamics matrix, then adding dynamics covaraince Q
	P_k_prior = np.dot(A, np.dot(P_km1, A.T)) + Q 

	#Perform causal intervation
	if refit_kf_inno_2:
		P_k_prior[:, :2] = 0.0
		P_k_prior[:2, :] = 0.0
		P_k_prior[4, :] = 0.0
		P_k_prior[:, 4] = 0.0

	"""
	2: Measurement update ==> incorporate the latest measurement z_k
	(Again, 'km1' means 'k minus 1')
	"""

	#Compute the ~Kalman gain~, which determines how new measurement is weighed against state estiamte
	#Note that this requires a matrix inversion, where the dimension of the matrix is the (big) observation dimension
	#This is computationally costly, and as a result this function can be slow
	K_k = np.dot(P_k_prior, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P_k_prior, H.T)) + R)))

	#Update state estimate
	x_hat_k = x_hat_k_prior + np.dot(K_k, z_k - np.dot(H, x_hat_k_prior))

	#Update state covariance
	P_k = np.dot((np.eye(A.shape[0]) - np.dot(K_k, H)), P_k_prior)

	return x_hat_k, P_k


"""
Run the Kalman filter by repeatedly applying the time-step update defined in 'kf_update'

Init. parameter inputs:
x_hat_0 = t=0 state mean
P_0     = t=0 state covaraince

System matrix inputs:
A = dynamics matrix 
H = observation matrix 
Q = dynamics noise 
R = observation noise 

Observation inputs:
Z = (time_steps, observation_dimension) observation timeseries

Optional inputs:
num_iter = number of iterations (defaults to len(Z))

Returns: 
x_hat = state mean timeseries (first entry will be x_hat_0)
"""
def run_kf(A, H, Q, R, Z, x_hat_0, P_0, refit_kf_inno_2=False, num_iter=None):
	if num_iter == None:
		num_iter = len(Z)
	x_hat = np.zeros((num_iter, len(x_hat_0)))
	x_hat[0] = x_hat_0
	P_k = P_0
	for k in range(1, num_iter):
		x_hat[k], P_k = kf_update(A, H, Q, R, P_k, x_hat[k-1], Z[k], refit_kf_inno_2)
	return x_hat


"""
Run the FULL Kalman smoother (forward and bcakward passes)

Init. parameter inputs:
x_hat_0 = t=0 state mean
P_0     = t=0 state covaraince

System matrix inputs:
A = dynamics matrix 
H = observation matrix 
Q = dynamics noise 
R = observation noise 

Observation inputs:
Z = (time_steps, observation_dimension) observation timeseries

Returns: 
x_hat = state mean timeseries
"""
def run_kf_smooth(A, H, Q, R, Z, x_hat_0, P_0):
	kf = pykalman.KalmanFilter(transition_matrices=A,
							   observation_matrices=H,
							   transition_covariance=Q,
							   observation_covariance=R,
							   initial_state_mean=x_hat_0,
							   initial_state_covariance=P_0)
	kf.smooth(Z)
	smooth_means, smooth_covs = kf.smooth(Z)
	return smooth_means


"""
Solve for the steady-state value of the A PRIORI value of P, the A PRIORI state covariance matrix (which converges independently of data)

For an explation of the DARE in the context of BMI, see [3] eq. 17.

For the scipy function which solves the DARE, see [4].

By comparing the notation in the two documnets, one sees that A and H need to be transposed 
before being passed into scipy.linalg.solve_discrete_are
Q, R are symmetric so transposing/not transposing is immaterial
"""
def steady_state_a_priori_P(A, H, Q, R, verbose=False):
	Q_prime = Q + np.eye(Q.shape[0])*10**-10
	R_prime = R + np.eye(R.shape[0])*10**-10
	return scipy.linalg.solve_discrete_are(A.T, H.T, Q_prime, R_prime)


"""
The steady-state KF has the form
x_{t+1} + A_sskf x_{t} + B_sskf y_{t+1}

This function returns A_sskf and B_sskf based on the KF system matrices A, H, Q, R, where 
A_sskf = (A - K H),
B_sskf = K,
where K is the steady-state Kalman gain computed using the steady-state a priori state covaraince.

See [3].

If refit_kf_inno_2 = True, then a modified version of the DARE is solved with 
A --> MA, Q --> MQM^T, where 
M = [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0]].
This corresponds to zeroing out non-velocity-velocity elements of the a priori state covariance matrix.
See [5].
Note that this setting assumes a 2D cursor scenario is being used.
"""
def steady_state_kf_matrices(A, H, Q, R, refit_kf_inno_2=False):

	if refit_kf_inno_2:
		M = np.zeros((5, 5))
		M[2:4, 2:4] = np.eye(2)
		P = steady_state_a_priori_P(np.dot(M, A), H, np.dot(M, np.dot(Q, M)), R)
	else:
		P = steady_state_a_priori_P(A, H, Q, R)

	K = np.dot(P, np.dot(H.T, np.linalg.inv(np.dot(H, np.dot(P, H.T)) + R))) #[3] eq. 21
	A_sskf = A - np.dot(A, np.dot(K, H))
	B_sskf = K
	return A_sskf, B_sskf

"""
Run the steady-state Kalman filter

A_sskf and B_sskf are as described above.
Measurements is a time-by-meas_dim structure of measurement vectors.
x_0 is the initial state estimate (set to zero if not provided).
NOTE: the first input measurement ends up being IGNORED! This is because the initial state estimate is some constant vector.
"""
def run_steady_state_kf(A_sskf, B_sskf, measurements, x_0=None):
	n = len(measurements)
	filtered = np.zeros((n, A_sskf.shape[0]))
	if type(x_0) == np.ndarray:
		filtered[0,:] = x_0
	for k in range(1, n):
		filtered[k,:] = np.dot(A_sskf, filtered[k-1,:]) + np.dot(B_sskf, measurements[k])
	return filtered


"""
Fit the system matrices A, H, Q, R (and x_0 and P_0) using MLE.
Note that this is easy since we know both the 'latent' and observed states. 
The MLE solutions for A and H coencide with their least-squares solutions.
After finding A and H, the covarainces Q and R may be calculated based on the emperical residuals of the mappings.
In this implementation, we allow for the X-Y data to be segmented into trials. This does not affect H and R, however the MLE
solutions for A and Q are a tiny bit different. The ability to fit A and H based on trial-segmented data is important for BMI
use cases in which data is often segmented into trials (e.g. reaches).

I worked out the equations myself for A and H (not hard).
The standard solutions for A, H, Q and R can found in e.g. [3].
"""
def fit_kf(X_trials, Y_trials):
	#Get A, H, Q, R
	A, Q = fit_transition_model(X_trials)
	H, R = fit_emission_model(X_trials, Y_trials)

	#Fit initial state and covariance
	firsts = np.asarray([trial[0] for trial in X_trials]).reshape((len(X_trials), X_trials[0].shape[1]))
	x_0 = np.mean(firsts, axis=0)
	P_0 = np.dot((firsts-x_0).T, firsts-x_0)

	return A, H, Q, R, x_0, P_0

"""
Subroutine of fit_kf (see fit_kf).
"""
def fit_transition_model(X_trials):
	#Fit transition matrix
	state_dim = X_trials[0].shape[1]
	M1 = np.zeros((state_dim, state_dim))
	M2 = np.zeros((state_dim, state_dim))
	for trial in X_trials:
		trial_without_last = trial[:-1,:]
		trial_without_first = trial[1:,:]
		M1 += np.dot(trial_without_first.T, trial_without_last)
		M2 += np.dot(trial_without_last.T, trial_without_last)
	A = np.dot(M1, np.linalg.inv(M2))

	#Fit transition covaraince
	Q = np.zeros((state_dim, state_dim))
	for trial in X_trials:
		trial_without_last = trial[:-1,:]
		trial_without_first = trial[1:,:]
		residuals = np.dot(trial_without_last, A.T) - trial_without_first
		Q += np.dot(residuals.T, residuals)
	Q = Q/(len(np.vstack(X_trials))-len(X_trials))

	return A, Q

"""
Subroutine of fit_kf (see fit_kf).
"""
def fit_emission_model(X_trials, Y_trials):
	#Smush trials together
	X = np.vstack(X_trials)
	Y = np.vstack(Y_trials)

	#Fit observation matrix and covariance
	H = np.dot(np.dot(Y.T, X), np.linalg.inv(np.dot(X.T, X)))
	residuals = np.dot(X, H.T) - Y
	R = np.dot(residuals.T, residuals)/len(X)

	return H, R

"""
Fit the matrices A, H, Q, R SUBJECT TO constraints on A and Q imposed in [5], namely,
A = [[1, 0, dt, 0, 0],
	 [0, 1, 0, dt, 0],
	 [0, 0, *,  *, 0],
	 [0, 0, *,  *, 0],
	 [0, 0, 0,  0, 1]]
Q = [[0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0],
	 [0, 0, *, *, 0],
	 [0, 0, *, *, 0],
	 [0, 0, 0, 0, 0]]
where the *'s can be anything.
This amounts to fitting an unconstrained model for the velocity data corresponding to the passed-in position data.
This function is designed to be used for fitting the KF params for a 2D (x-y) cursor (and follows the work in [5], without any of the modifications).
"""
def fit_kf_2d_cursor(curosr_position_trials, neural_obs_trials, delta_t):

	#Calculate velocity trials
	velocity_trials = [(x[1:]-x[:-1])/delta_t for x in curosr_position_trials]

	#Form trials of the form (position, velocity, 1)
	kinematics_trials = [ np.hstack((curosr_position_trials[i][:-1], velocity_trials[i], np.ones( (len(velocity_trials[i]), 1) ))) for i in range(len(curosr_position_trials))]
	
	#One time-step is lost when calculating velocity for trials
	neural_obs_trials = [neural_obs_trials[i][:-1] for i in range(len(neural_obs_trials))]

	#Form constrained transition/transition noise matrices
	vel_trans_mat, vel_cov_mat = fit_transition_model(velocity_trials)

	A = np.zeros((5, 5))
	A[:2, :2] = np.eye(2)
	A[:2, 2:4] = np.eye(2)*delta_t
	A[2:4, 2:4] = vel_trans_mat
	A[4, 4] = 1.0

	Q = np.zeros((5, 5))
	Q[2:4, 2:4] = vel_cov_mat

	#Get emission/emission noise matrices
	H, R = fit_emission_model(kinematics_trials, neural_obs_trials)

	return A, H, Q, R

"""
Fit the matrices A, H, Q, R SUBJECT TO constraints on A and Q imposed in [5], BUT for a
1D cursor (vocal pitch, in my case) as opposed to a 2D cursor:
A = [[1, dt, 0],
	 [0,  *, 0],
	 [0,  0, 1]]
Q = [[0, 0, 0],
	 [0, *, 0],
	 [0, 0, 0]]
where the *'s can be anything.
"""
def fit_kf_1d_cursor(pitch_trials, neural_obs_trials, delta_t):

	#These will be 1D, so reshape them to be (N, 1) so they place nicely with the functions here
	pitch_trials = [p.reshape(len(p), 1) for p in pitch_trials]

	#Calculate velocity trials
	velocity_trials = [(x[1:]-x[:-1])/delta_t for x in pitch_trials]

	#Form trials of the form (position, velocity, 1)
	kinematics_trials = [ np.hstack((pitch_trials[i][:-1], velocity_trials[i], np.ones( (len(velocity_trials[i]), 1) ))) for i in range(len(pitch_trials))]
	
	#One time-step is lost when calculating velocity for trials
	neural_obs_trials = [neural_obs_trials[i][:-1] for i in range(len(neural_obs_trials))]

	#Form constrained transition/transition noise matrices
	#NOTE: these will just be scalars in this case! (Actually, 1x1 matrices...)
	vel_trans_mat, vel_cov_mat = fit_transition_model(velocity_trials)

	A = np.zeros((3, 3))
	A[0, 0] = 1.0
	A[0, 1] = delta_t
	A[1, 1] = vel_trans_mat
	A[2, 2] = 1.0

	Q = np.zeros((3, 3))
	Q[1, 1] = vel_cov_mat

	#Get emission/emission noise matrices
	H, R = fit_emission_model(kinematics_trials, neural_obs_trials)

	return A, H, Q, R



