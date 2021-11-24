import numpy as np
import scipy.linalg  # You may find scipy.linalg.block_diag useful
import scipy.stats  # You may find scipy.stats.multivariate_normal.pdf useful
from . import turtlebot_model as tb

EPSILON_OMEGA = 1e-3

class ParticleFilter(object):
    """
    Base class for Monte Carlo localization and FastSLAM.

    Usage:
        pf = ParticleFilter(x0, R)
        while True:
            pf.transition_update(u, dt)
            pf.measurement_update(z, Q)
            localized_state = pf.x
    """

    def __init__(self, x0, R):
        """
        ParticleFilter constructor.

        Inputs:
            x0: np.array[M,3] - initial particle states.
             R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
        """
        self.M = x0.shape[0]  # Number of particles
        self.xs = x0  # Particle set [M x 3]
        self.ws = np.repeat(1. / self.M, self.M)  # Particle weights (initialize to uniform) [M]
        self.R = R  # Control noise covariance (corresponding to dt = 1 second) [2 x 2]

    @property
    def x(self):
        """
        Returns the particle with the maximum weight for visualization.

        Output:
            x: np.array[3,] - particle with the maximum weight.
        """
        idx = self.ws == self.ws.max()
        x = np.zeros(self.xs.shape[1:])
        x[:2] = self.xs[idx,:2].mean(axis=0)
        th = self.xs[idx,2]
        x[2] = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        return x

    def transition_update(self, u, dt):
        """
        Performs the transition update step by updating self.xs.

        Inputs:
            u: np.array[2,] - zero-order hold control input.
            dt: float        - duration of discrete time step.
        Output:
            None - internal belief state (self.xs) should be updated.
        """
        ########## Code starts here ##########
        # DONE: Update self.xs.
        # Hint: Call self.transition_model().
        # Hint: You may find np.random.multivariate_normal useful.
        epsilon = np.random.multivariate_normal(mean=np.zeros_like(u),
                                                cov=self.R * dt,
                                                size=self.M)
        us = np.array([u,] * self.M) + epsilon
        self.xs = self.transition_model(us, dt)
        ########## Code ends here ##########

    def transition_model(self, us, dt):
        """
        Propagates exact (nonlinear) state dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """
        raise NotImplementedError("transition_model must be overridden by a subclass of EKF")

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        raise NotImplementedError("measurement_update must be overridden by a subclass of EKF")

    def resample(self, xs, ws):
        """
        Resamples the particles according to the updated particle weights.
        Low variance re-sample strategy

        Inputs:
            xs: np.array[M,3] - matrix of particle states.
            ws: np.array[M,]  - particle weights.

        Output:
            None - internal belief state (self.xs, self.ws) should be updated.
        """
        r = np.random.rand() / self.M

        ########## Code starts here ##########
        # DONE: Update self.xs, self.ws.
        # Note: Assign the weights in self.ws to the corresponding weights in ws
        #       when resampling xs instead of resetting them to a uniform
        #       distribution. This allows us to keep track of the most likely
        #       particle and use it to visualize the robot's pose with self.x.
        # Hint: To maximize speed, try to implement the resampling algorithm
        #       without for loops. You may find np.linspace(), np.cumsum(), and
        #       np.searchsorted() useful. This results in a ~10x speedup.
        cum_ws = np.cumsum(ws)
        m = np.linspace(start=0, stop=self.M, num=self.M, endpoint=False) / float(self.M)
        selected_idx = np.searchsorted(cum_ws,
                                   cum_ws[-1] * (r + m))
        self.xs = xs[selected_idx, :]
        self.ws = ws[selected_idx]
        ########## Code ends here ##########

    def measurement_model(self, z_raw, Q_raw):
        """
        Converts raw measurements into the relevant Gaussian form (e.g., a
        dimensionality reduction).

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[2I,]   - joint measurement mean.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        raise NotImplementedError("measurement_model must be overridden by a subclass of EKF")


class MonteCarloLocalization(ParticleFilter):

    def __init__(self, x0, R, map_lines, tf_base_to_camera, g):
        """
        MonteCarloLocalization constructor.

        Inputs:
                       x0: np.array[M,3] - initial particle states.
                        R: np.array[2,2] - control noise covariance (corresponding to dt = 1 second).
                map_lines: np.array[2,J] - J map lines in columns representing (alpha, r).
        tf_base_to_camera: np.array[3,]  - (x, y, theta) transform from the
                                           robot base to camera frame.
                        g: float         - validation gate.
        """
        self.map_lines = map_lines  # Matrix of J map lines with (alpha, r) as columns
        self.tf_base_to_camera = tf_base_to_camera  # (x, y, theta) transform
        self.g = g  # Validation gate
        super(self.__class__, self).__init__(x0, R)

    def transition_model(self, us, dt):
        """
        Unicycle model dynamics.

        Inputs:
            us: np.array[M,2] - zero-order hold control input for each particle.
            dt: float         - duration of discrete time step.
        Output:
            g: np.array[M,3] - result of belief mean for each particle
                               propagated according to the system dynamics with
                               control u for dt seconds.
        """

        ########## Code starts here ##########
        # DONE: Compute g.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: A simple solution can be using a for loop for each partical
        #       and a call to tb.compute_dynamics
        # Hint: To maximize speed, try to compute the dynamics without looping
        #       over the particles. If you do this, you should implement
        #       vectorized versions of the dynamics computations directly here
        #       (instead of modifying turtlebot_model). This results in a
        #       ~10x speedup.
        # Hint: This faster/better solution does not use loop and does 
        #       not call tb.compute_dynamics. You need to compute the idxs
        #       where abs(om) > EPSILON_OMEGA and the other idxs, then do separate 
        #       updates for them


        ########## Code ends here ##########
        def regular_update(us, dt):
            x_prev, y_prev, th_prev = self.xs[:, 0], self.xs[:, 1], self.xs[:, 2]
            V, om = us[:, 0], us[:, 1]

            th_now = th_prev + om * dt
            sin_th_prev, sin_th_now = np.sin(th_prev), np.sin(th_now)
            cos_th_prev, cos_th_now = np.cos(th_prev), np.cos(th_now)
            V_over_om = V / np.maximum(np.abs(om), EPSILON_OMEGA) * np.sign(om)

            x_now = x_prev + V_over_om * (+sin_th_now - sin_th_prev)
            y_now = y_prev + V_over_om * (-cos_th_now + cos_th_prev)

            return np.stack([x_now, y_now, th_now], axis=-1)

        def small_om_update(us, dt):
            mid_sin = (np.sin(th_prev) + np.sin(th_now)) / 2.
            mid_cos = (np.cos(th_prev) + np.cos(th_now)) / 2.

            x_now = x_prev + V * mid_cos * dt
            y_now = y_prev + V * mid_sin * dt

            return np.stack([x_now, y_now, th_now], axis=-1)

        x_prev, y_prev, th_prev = self.xs[:, 0], self.xs[:, 1], self.xs[:, 2]
        V, om = us[:, 0], us[:, 1]
        th_now = th_prev + om * dt

        g = np.where(np.expand_dims(np.abs(us[:, 1]), axis=1) < EPSILON_OMEGA,
                     small_om_update(us, dt),
                     regular_update(us, dt))
        return g

    def measurement_update(self, z_raw, Q_raw):
        """
        Updates belief state according to the given measurement.

        Inputs:
            z_raw: np.array[2,I]   - matrix of I columns containing (alpha, r)
                                     for each line extracted from the scanner
                                     data in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Output:
            None - internal belief state (self.x, self.ws) is updated in self.resample().
        """
        xs = np.copy(self.xs)
        # ws = np.zeros_like(self.ws)

        ########## Code starts here ##########
        # DONE: Compute new particles (xs, ws) with updated measurement weights.
        # Hint: To maximize speed, implement this without looping over the
        #       particles. You may find scipy.stats.multivariate_normal.pdf()
        #       useful.
        # Hint: You'll need to call self.measurement_model()
        vs, Q = self. measurement_model(z_raw, Q_raw)
        ws = scipy.stats.multivariate_normal.pdf(vs, mean=None, cov=Q)
        ########## Code ends here ##########

        self.resample(xs, ws)

    def measurement_model(self, z_raw, Q_raw):
        """
        Assemble one joint measurement and covariance from the individual values
        corresponding to each matched line feature for each particle.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: [np.array[2,2]] - list of I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            z: np.array[M,2I]  - joint measurement mean for M particles.
            Q: np.array[2I,2I] - joint measurement covariance.
        """
        vs = self.compute_innovations(z_raw, np.array(Q_raw))

        ########## Code starts here ##########
        # DONE: Compute Q.
        # Hint: You might find scipy.linalg.block_diag() useful
        Q = scipy.linalg.block_diag(*Q_raw)
        ########## Code ends here ##########

        return vs, Q

    def compute_innovations(self, z_raw, Q_raw):
        """
        Given lines extracted from the scanner data, tries to associate each one
        to the closest map entry measured by Mahalanobis distance.

        Inputs:
            z_raw: np.array[2,I]   - I lines extracted from scanner data in
                                     columns representing (alpha, r) in the scanner frame.
            Q_raw: np.array[I,2,2] - I covariance matrices corresponding
                                     to each (alpha, r) column of z_raw.
        Outputs:
            vs: np.array[M,2I] - M innovation vectors of size 2I
                                 (predicted map measurement - scanner measurement).
        """
        def angle_diff(a, b):
            a = a % (2. * np.pi)
            b = b % (2. * np.pi)
            diff = a - b
            if np.size(diff) == 1:
                if np.abs(a - b) > np.pi:
                    sign = 2. * (diff < 0.) - 1.
                    diff += sign * 2. * np.pi
            else:
                idx = np.abs(diff) > np.pi
                sign = 2. * (diff[idx] < 0.) - 1.
                diff[idx] += sign * 2. * np.pi
            return diff

        ########## Code starts here ##########
        # DONE: Compute vs (with shape [M x I x 2]).
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       observed line, find the most likely map entry (the entry with 
        #       least Mahalanobis distance).
        # Hint: To maximize speed, try to eliminate all for loops, or at least
        #       for loops over J. It is possible to solve multiple systems with
        #       np.linalg.solve() and swap arbitrary axes with np.transpose().
        #       Eliminating loops over J results in a ~10x speedup.
        #       Eliminating loops over I results in a ~2x speedup.
        #       Eliminating loops over M results in a ~5x speedup.
        #       Overall, that's 100x!
        # Hint: For the faster solution, you might find np.expand_dims(), 
        #       np.linalg.solve(), np.meshgrid() useful.
        hs = self.compute_predicted_measurements()  # (M, 2, J)
        hs = np.transpose(hs, (0, 2, 1))    # (M, J, 2)
        hs = np.expand_dims(hs, 1)  # (M, 1, J, 2)

        z_raw = z_raw.T # (I, 2)
        z_raw = np.expand_dims(z_raw, 0)
        z_raw = np.expand_dims(z_raw, 2)    # (1, I, 1, 2)

        vs = z_raw - hs     # (M, I, J, 2)
        vs_q = np.matmul(vs, np.linalg.inv(Q_raw))  # (M, I, J, 2)
        d_sq = np.sum(vs_q * vs, axis=-1)   # (M, I, J)

        min_index = np.argmin(d_sq, axis=-1)    # (M, I)
        min_index = np.expand_dims(min_index, -1)   # (M, I, 1)
        min_index = np.expand_dims(min_index, -1)   # (M, I, 1, 1)

        vs = np.squeeze(np.take_along_axis(vs, min_index, axis=2))  # (M, I, 2)
        ########## Code ends here ##########

        # Reshape [M x I x 2] array to [M x 2I]
        return vs.reshape((self.M,-1))  # [M x 2I]

    def compute_predicted_measurements(self):
        """
        Given a single map line in the world frame, outputs the line parameters
        in the scanner frame so it can be associated with the lines extracted
        from the scanner measurements.

        Input:
            None
        Output:
            hs: np.array[M,2,J] - J line parameters in the scanner (camera) frame for M particles.
        """
        ########## Code starts here ##########
        # DONE: Compute hs.
        # Hint: We don't need Jacobians for particle filtering.
        # Hint: Simple solutions: Using for loop, for each particle, for each 
        #       map line, transform to scanner frmae using tb.transform_line_to_scanner_frame()
        #       and tb.normalize_line_parameters()
        # Hint: To maximize speed, try to compute the predicted measurements
        #       without looping over the map lines. You can implement vectorized
        #       versions of turtlebot_model functions directly here. This
        #       results in a ~10x speedup.
        # Hint: For the faster solution, it does not call tb.transform_line_to_scanner_frame()
        #       or tb.normalize_line_parameters(), but reimplement these steps vectorized.
        alpha, r = self.map_lines[0, :], self.map_lines[1, :]   # column vectors (J, )
        x_base, y_base, th_base = np.expand_dims(self.xs[:, 0], -1), \
                                  np.expand_dims(self.xs[:, 1], -1), \
                                  np.expand_dims(self.xs[:, 2], -1)     # (M, 1)
        x_camera_base, y_camera_base, th_camera_base = self.tf_base_to_camera   # float
        # Rotation matrix to get camera coordinates in
        x_camera = x_base + np.cos(th_base) * x_camera_base - np.sin(th_base) * y_camera_base    # (M, 1)
        y_camera = y_base + np.sin(th_base) * x_camera_base + np.cos(th_base) * y_camera_base    # (M, 1)

        alpha_c = alpha - th_base - th_camera_base  # (M, J)
        r_c = r - x_camera * np.cos(alpha) - y_camera * np.sin(alpha)   # (M, J)

        # Normalize line parameters
        alpha_c = np.where(r_c < 0, alpha_c + np.pi, alpha_c)
        r_c = np.abs(r_c)
        alpha_c = (alpha_c + np.pi) % (2 * np.pi) - np.pi

        hs = np.hstack([alpha_c, r_c])  # (M, 2J)
        hs = np.reshape(hs, (self.M, 2, -1))    # (M, 2, J)
        ########## Code ends here ##########

        return hs

