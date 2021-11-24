import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # DONE: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    V, om = u
    x_prev, y_prev, th_prev = xvec
    th_now = th_prev + om * dt

    if abs(om) < EPSILON_OMEGA:
        # omega is close to 0
        # uses mid point rule to compute dynamic update
        mid_sin = (np.sin(th_prev) + np.sin(th_now)) / 2.
        mid_cos = (np.cos(th_prev) + np.cos(th_now)) / 2.
        x_now = x_prev + mid_cos * V * dt
        y_now = y_prev + mid_sin * V * dt
        if compute_jacobians:
            Gx = np.array([[1, 0, -V * mid_sin * dt],
                           [0, 1,  V * mid_cos * dt],
                           [0, 0, 1], ])
            Gu = np.array([[mid_cos * dt, -V / 2. * np.sin(th_now) * dt * dt],
                           [mid_sin * dt,  V / 2. * np.cos(th_now) * dt * dt],
                           [0, dt], ])
        else:
            Gx = Gu = None
    else:
        # omega is large
        sin_th_now = np.sin(th_now)
        sin_th_prev = np.sin(th_prev)
        cos_th_now = np.cos(th_now)
        cos_th_prev = np.cos(th_prev)

        om_inv = 1. / om
        V_over_om = V / om

        x_now = x_prev + V_over_om * (+sin_th_now - sin_th_prev)
        y_now = y_prev + V_over_om * (-cos_th_now + cos_th_prev)

        if compute_jacobians:
            Gx = np.array([[1, 0, V_over_om * (cos_th_now - cos_th_prev)],
                           [0, 1, V_over_om * (sin_th_now - sin_th_prev)],
                           [0, 0, 1], ])
            Gu = np.array([[om_inv * (+sin_th_now - sin_th_prev),
                                V_over_om * cos_th_now * dt - V_over_om * om_inv * (+sin_th_now - sin_th_prev)],
                           [om_inv * (-cos_th_now + cos_th_prev),
                                V_over_om * sin_th_now * dt - V_over_om * om_inv * (-cos_th_now + cos_th_prev)],
                           [0, dt], ])
        else:
            Gx = Gu = None

    g = np.array([x_now, y_now, th_now])
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    x_world, y_world, th_world  = x
    rotation_matrix = np.array([[np.cos(th_world), -np.sin(th_world), 0],
                                [np.sin(th_world), np.cos(th_world), 0],
                                [0, 0, 1]])

    # pose of camera in world frame
    x_cam, y_cam, th_cam = x + np.dot(rotation_matrix, tf_base_to_camera)
    # line parameter in camera frame, new alpha and new r
    h = np.array([alpha - th_cam,
                  r - x_cam * np.cos(alpha) - y_cam * np.sin(alpha)])

    if compute_jacobian:
        x_base_cam, y_base_cam, _ = tf_base_to_camera
        # recall that x_cam = x_x + np.cos(th_world) * x_base_cam - np.sin(th_world) * y_base_cam
        # recall that y_cam = x_y + np.sin(th_world) * x_base_cam + np.cos(th_world) * y_base_cam
        Hx = np.array([[0, 0, -1],
                       [-np.cos(alpha), -np.sin(alpha),
                        -np.cos(alpha) * (-np.sin(th_world) * x_base_cam - np.cos(th_world) * y_base_cam)
                        -np.sin(alpha) * (+np.cos(th_world) * x_base_cam - np.sin(th_world) * y_base_cam)], ])
    else:
        Hx = None
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
