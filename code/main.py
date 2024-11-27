import rclpy, time
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from code.utils.GeneratorNode      import GeneratorNode
from code.utils.TransformHelpers   import *
from code.utils.TrajectoryUtils    import *

# Grab the general fkin from HW6 P1.
from code.utils.KinematicChain     import KinematicChain

# Import the ball
from code.Balls import Balls


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain = KinematicChain(node, 'world', 'lightsaber_blade', self.jointnames())

        self.q0 = np.radians(np.array([0, 0, 0, 0, 0, 0, 0]))
        self.p0, self.R0, _, _ = self.chain.fkin(self.q0)

        self.qd = self.q0
        self.lam = 10
        self.pos = self.get_balls_info()[0][:3] + self.get_balls_info()[0][3:]
        self.r0 = self.dist_p_to_seg(self.pos, self.p0, self.p0 + self.R0 @ np.array([0, 0, 0.8]))
        self.blade_length = 0.8

        self.pos2 = self.get_balls_info()[1][:3] + self.get_balls_info()[1][3:]

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7']

    @staticmethod
    def inv(J, lam):
        return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(J.shape[0]))
    
    @staticmethod
    def get_balls_info():
        balls = Balls.get_balls()
        return [np.concatenate((ball['p'], ball['v'])) for ball in balls]

    @staticmethod
    def dist_p_to_seg(P, A, B):
        AP = P - A
        AB = B - A
        t = max(0, min(1, np.dot(AP, AB) / np.dot(AB, AB)))
        return np.linalg.norm(P - (A + t * AB))

    def JFull(self, q, x):
        ptip, Rtip, Jv, Jw = self.chain.fkin(q)
        L = self.blade_length
        J3 = np.vstack((Rtip.T @ Jv, Rtip.T @ Jw))
        J2 = np.array([
            [1, 0, 0,  0, 0, 0],
            [0, 1, 0,  0, 0, 0],
            [0, 0, 1,  0, 0, 0],
            [1, 0, 0,  0, L, 0],
            [0, 1, 0, -L, 0, 0],
            [0, 0, 1,  0, 0, 0]
        ])
        T0T = T_from_Rp(Rtip, ptip)
        xt = p_from_T(np.linalg.inv(T0T) @ T_from_Rp(Reye(), x))
        dx = 0.00001
        A = np.array([0, 0, 0])
        B = np.array([0, 0, L])
        dist = self.dist_p_to_seg(xt, A, B)
        J1 = np.array([
            (self.dist_p_to_seg(xt, A + np.array([dx, 0, 0]), B) - dist) / dx,
            (self.dist_p_to_seg(xt, A + np.array([0, dx, 0]), B) - dist) / dx,
            (self.dist_p_to_seg(xt, A + np.array([0, 0, dx]), B) - dist) / dx,
            (self.dist_p_to_seg(xt, A, B + np.array([dx, 0, 0])) - dist) / dx,
            (self.dist_p_to_seg(xt, A, B + np.array([0, dx, 0])) - dist) / dx,
            (self.dist_p_to_seg(xt, A, B + np.array([0, 0, dx])) - dist) / dx
        ])
        return J1 @ J2 @ J3

        

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        # if (t < 1):
        #     pd, vd = goto(t, 1, self.p0, self.pos)
        #     Rd = Reye()
        #     wd = pzero()
        # else:
        #     pd = self.get_balls_info()[0][:3]
        #     vd = self.get_balls_info()[0][3:]
        #     Rd = Reye()
        #     wd = pzero()

        # qdlast = self.qd
        
        # ptip, Rtip, Jv, Jw = self.chain.fkin(qdlast)
        # xr = np.concatenate((vd, wd)) + self.lam * np.concatenate((ep(pd, ptip), eR(Rd, Rtip)))
        # qddot = self.inv(np.vstack((Jv, Jw)), 0.1) @ xr
        # qd = qdlast + (qddot * dt)
        
        # self.qd = qd

        if (t < 2):
            rd, rdotd = goto(t, 2, self.r0, 0)
        else:
            rd = 0
            rdotd = 0
        
        qdlast = self.qd
        J = self.JFull(qdlast, self.pos)
        ptip, Rtip, _, _ = self.chain.fkin(qdlast)
        xr = rdotd + self.lam * (rd - self.dist_p_to_seg(self.pos, ptip, ptip + Rtip @ np.array([0, 0, 0.8])))
        qddot = self.inv(J, 1) * xr + ((np.eye(7) - self.inv(J, 1) @ J) @ self.inv(self.JFull(qdlast, self.pos2), 1) * -0.5)
        qd = qdlast + (qddot * dt)

        self.qd = qd

        pd = pzero()
        vd = pzero()
        Rd = Reye()
        wd = pzero()
        
        # Return the desired joint and task (position/orientation) pos/vel.
        return (qd, qddot, pd, vd, Rd, wd)


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
    balls = Balls('balls', UPDATE_RATE)
    balls.add_ball(np.array([0.3, 1, 0.3]), np.array([0, 0, 0]))
    balls.add_ball(np.array([0.3, 0.5, 0.3]), np.array([0, 0, 0]))
    SPIN_QUEUE.append(balls)

    generator = GeneratorNode('generator', UPDATE_RATE, Trajectory)
    SPIN_QUEUE.append(generator)

    while SPIN_QUEUE:
        start_time = time.time()
        for node in SPIN_QUEUE:
            rclpy.spin_once(node, timeout_sec=1/(UPDATE_RATE))
        SPIN_QUEUE = [node for node in SPIN_QUEUE if not hasattr(node, 'future') or not node.future.done()]
        timeout = max(0, 1/UPDATE_RATE - (time.time() - start_time))
        if timeout > 0:
            time.sleep(timeout)
        else:
            print(f"Warning: loop took longer than {1000/UPDATE_RATE} ms, took {(time.time() - start_time) * 1000:.3f} ms")

    # Shutdown the node and ROS.
    generator.shutdown()
    balls.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
