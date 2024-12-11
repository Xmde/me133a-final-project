import rclpy, time
import numpy as np
import traceback

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from code.utils.GeneratorNode import GeneratorNode
from code.utils.TransformHelpers import *
from code.utils.TrajectoryUtils import *

# Grab the general fkin from HW6 P1.
from code.utils.KinematicChain import KinematicChain

# Import the ball
from code.Balls import Balls


#
#   Trajectory Class
#
class Trajectory:
    # Initialization.
    def __init__(self, node):
        self.node = node

        # Set up the kinematic chain object.
        self.full_chain = KinematicChain(
            node, "world", "lightsaber_blade_end", self.jointnames()
        )
        self.blade_rot_chain = KinematicChain(
            node, "world", "lightsaber_blade_tilt", self.jointnames()[:-1]
        )

        self.q0 = np.array([0, 0, 0, 0, 0, 0, 0, 0.4, np.pi / 2, np.pi / 2, 0])
        self.p0, self.R0, _, _ = self.full_chain.fkin(self.q0)

        self.qd = self.q0.copy()
        self.lam = 20
        self.angle_lam = 1
        self.lam_perp = 20
        self.lam_dist = 100
        self.blade_len_max = 0.75
        self.blade_len_min = 0.05

        self.tracking = False
        self.traj_start = 0
        self.traj_time = 1.5
        self.traj_p0 = self.p0
        self.traj_pf = None
        self.traj_dangle0 = self.compute_angle(self.R0[:, 2])

    def jointnames(self):
        """
        Declare the joint names.

        Returns:
            list: A list of joint names for the expected URDF.
        """
        return [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
            "light_saber_blade_len",
            "light_saber_blade_pan",
            "light_saber_blade_tilt",
            "light_saber_blade_end",
        ]

    @staticmethod
    def winv(J, W, gamma=1e-3):
        """
        Compute the weighted pseudoinverse of a matrix.

        Args:
            J (np.ndarray): Jacobian matrix.
            W (np.ndarray): Weighting matrix.

        Returns:
            np.ndarray: Weighted pseudoinverse of J.
        """
        Jw = J @ W
        
        return W @ np.linalg.inv(Jw.T @ Jw + (gamma ** 2) * np.eye(Jw.shape[1])) @ Jw.T

    @staticmethod
    def get_balls_info():
        """
        Get information about the balls in the scene.

        Returns:
            list: List of position and velocity vectors for each ball.
        """
        balls = Balls.get_balls()
        return [np.concatenate((ball["p"], ball["v"])) for ball in balls]
    
    @staticmethod
    def get_balls_interval():
        """
        Get the time interval for the balls in the scene.

        Returns:
            float: Time interval for the balls.
        """
        balls = Balls.get_balls()
        return balls[1]["spawn_time"] - balls[0]["spawn_time"]

    def compute_angle(self, vec):
        """
        Compute the pan and tilt angles required to align the blade with a given vector.

        Args:
            vec (np.ndarray): A 3D vector.

        Returns:
            np.ndarray: Pan and tilt angles.
        """
        # Ensure vec is a unit vector
        vec_norm = np.linalg.norm(vec)
        if np.isclose(vec_norm, 0):
            self.node.get_logger().info(
                "Input vector has zero length in compute_angle."
            )
            vec = np.array([0, 0, 1])  # Default to a unit vector along Z-axis
            vec_norm = 1.0
        vec = vec / vec_norm

        sin_theta = vec[2]
        cos_theta = np.sqrt(1 - sin_theta**2)
        # Prevent division by zero
        if np.isclose(cos_theta, 0):
            cos_theta = np.finfo(float).eps
        pan = np.arctan2(-vec[0] / cos_theta, vec[1] / cos_theta)
        tilt = np.arcsin(sin_theta)
        return np.array([pan, tilt])

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        """
        Evaluate the trajectory at a given time.

        Args:
            t (float): Current time.
            dt (float): Time step.

        Returns:
            tuple: Desired joint positions and velocities, task space positions and velocities.
        """
        try:
            # Get ball information
            ball_info = self.get_balls_info()[0]
            ball_pos = ball_info[:3]
            ball_vel = ball_info[3:]
            ball_vel_norm = np.linalg.norm(ball_vel)
            # print(f"Interval: {self.get_balls_interval()}")

            if np.isclose(ball_vel_norm, 0):
                self.node.get_logger().info(
                    "Ball velocity norm is zero, setting default velocity."
                )
                ball_vel = np.array([0, -1, 0])
                ball_vel_norm = np.linalg.norm(ball_vel)

            W = np.diag([1, 1, 1, 1, 1, 1, 1, 0.5, 5, 5, 10])

            max_tracking_time = 5.0  # seconds
            max_tracking_distance = 3.0  # meters
            distance_to_ball = np.linalg.norm(ball_pos - self.traj_p0)

            if not self.tracking:
                if self.traj_pf is None:
                    self.traj_pf = ball_pos + ball_vel * self.traj_time
                pos, vel = spline(
                    t - self.traj_start,
                    self.traj_time,
                    self.traj_p0,
                    self.traj_pf,
                    pzero(),
                    ball_vel,
                )
                dangle, danglevel = spline(
                    t - self.traj_start,
                    self.traj_time,
                    self.traj_dangle0,
                    self.compute_angle(ball_vel / -ball_vel_norm),
                    np.zeros(2),
                    np.zeros(2),
                )
                if t - self.traj_start >= self.traj_time:
                    self.tracking = True
                    self.tracking_start_time = t
            else:
                # Reset tracking if maximum time or distance is exceeded
                if (
                    t - self.tracking_start_time
                ) > max_tracking_time or distance_to_ball > max_tracking_distance:
                    self.node.get_logger().info(
                        "Tracking timeout or ball out of range. Resetting trajectory."
                    )
                    Balls.cycle_first_ball(Balls.gen_random_posvel(10))
                    self.tracking = False
                    self.traj_start = t
                    self.traj_p0 = self.p0
                    self.traj_pf = None
                    return self.evaluate(t, dt)  # Re-evaluate with the next ball

                pos = ball_pos
                vel = ball_vel

                dangle = self.compute_angle(ball_vel / -ball_vel_norm)
                danglevel = np.zeros(2)

            qd_last = self.qd.copy()

            # Forward kinematics
            _, Rrot, _, Jwrot = self.blade_rot_chain.fkin(qd_last[:-1])
            Jwrot = np.array([[0, 0, 1], [1, 0, 0]]) @ np.hstack(
                (Jwrot, np.zeros((3, 1)))
            )
            ptip, Rtip, Jvtip, _ = self.full_chain.fkin(qd_last)

            recompute = True
            while recompute:
                # Enforce blade length limits (BAD IMPLEMENTATION)
                # if qd_last[7] > self.blade_len_max or qd_last[7] < self.blade_len_min:
                #     Jvtip[:, 7] = np.zeros(3)
                #     qd_last[7] = np.clip(qd_last[7], self.blade_len_min, self.blade_len_max)
                # if (qd_last[10] < -0.03):
                #     Jwrot[:, 10] = np.zeros(2)
                #     Jvtip[:, 10] = np.zeros(3)
                #     qd_last[10] = -0.03

                # Control laws
                angle_error = dangle - self.compute_angle(Rrot[:, 2])
                qddot = self.winv(Jwrot, W) @ (danglevel + (self.angle_lam * angle_error))

                null_space_projection = np.eye(11) - self.winv(Jwrot, W) @ Jwrot
                position_error = vel + (self.lam * ep(pos, ptip))
                qddot += null_space_projection @ (self.winv(Jvtip, W) @ position_error)

                # Enforce secondary objectives
                sec_error = self.lam_perp * np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2 - qd_last[9], 0]
                )
                # sec_error += self.lam_dist * np.array(
                #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -qd_last[10]]
                # )
                qddot += (
                    null_space_projection
                    @ (np.eye(11) - self.winv(Jvtip, W) @ Jvtip)
                    @ sec_error
                )

                # Enforce joint limits (GOOD IMPLEMENTATION)
                self.qd = qd_last + qddot * dt
                recompute = False
                if self.qd[7] > self.blade_len_max and qddot[7] > 0:
                    Jwrot[:, 7] = np.zeros(2)
                    Jvtip[:, 7] = np.zeros(3)
                    recompute = True
                if self.qd[7] < self.blade_len_min and qddot[7] < 0:
                    Jwrot[:, 7] = np.zeros(2)
                    Jvtip[:, 7] = np.zeros(3)
                    recompute = True
                if self.qd[10] < -0.03 and qddot[10] < 0:
                    Jwrot[:, 10] = np.zeros(2)
                    Jvtip[:, 10] = np.zeros(3)
                    recompute = True

            
            # Update joint positions
            self.qd[7] = np.clip(self.qd[7], self.blade_len_min, self.blade_len_max)

            collision_distance = np.linalg.norm(ep(ball_pos, ptip))
            self.node.get_logger().debug(f"Collision distance: {collision_distance}")

            # Check for collision and reset trajectory
            if self.qd[-1] < 0 and collision_distance < 0.1:
                self.node.get_logger().info(
                    "Collision detected. Cycling and resetting trajectory."
                )
                # self.traj_time = self.get_balls_interval() / 2
                Balls.cycle_first_ball(Balls.gen_random_posvel(10))
                self.tracking = False
                self.traj_start = t
                self.traj_p0 = ptip
                self.traj_pf = None
                self.traj_dangle0 = self.compute_angle(Rtip[:, 2])

            pd = ptip
            vd = pzero()
            Rd = Rtip
            wd = pzero()

            # Return the desired joint and task space positions and velocities
            return (self.qd, qddot, pd, vd, Rd, wd)
        except Exception as e:
            self.node.get_logger().info(f"Error in evaluating trajectory: {e}")
            traceback.print_exc()
            return None


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)
    SPIN_QUEUE = []
    UPDATE_RATE = 100

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    balls = Balls("balls", UPDATE_RATE)
    # Makes two balls and puts them randomly in the scene.
    NUM_BALLS = 4
    for i in range(NUM_BALLS):
        balls.add_ball(Balls.gen_random_posvel((10.0 / NUM_BALLS) * (i + 1)))
    SPIN_QUEUE.append(balls)

    generator = GeneratorNode("generator", UPDATE_RATE, Trajectory)
    SPIN_QUEUE.append(generator)

    while SPIN_QUEUE:
        start_time = time.time()
        for node in SPIN_QUEUE:
            rclpy.spin_once(node, timeout_sec=1 / (UPDATE_RATE))
        SPIN_QUEUE = [
            node
            for node in SPIN_QUEUE
            if not hasattr(node, "future") or not node.future.done()
        ]
        timeout = max(0, 1 / UPDATE_RATE - (time.time() - start_time))
        if timeout > 0:
            time.sleep(timeout)
        else:
            print(
                f"Warning: loop took longer than {1000/UPDATE_RATE} ms, took {(time.time() - start_time) * 1000:.3f} ms"
            )

    # Shutdown the node and ROS.
    generator.shutdown()
    balls.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
