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
        self.lam = 30
        self.blade_length = 0.7
        self.blade_guard_offset = 0.05
        self.init_balls_info = self.get_balls_info()
        self.r0 = self.dist_p_to_seg(self.init_balls_info[0][:3] + 5 * self.init_balls_info[0][3:], self.p0 + self.R0 @ np.array([0, 0, self.blade_guard_offset]), self.p0 + self.R0 @ np.array([0, 0, self.blade_length]))

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7']

    # We use weighted inverse becuase there seems to be a lot of singularities.
    @staticmethod
    def inv(J, lam):
        return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(J.shape[0]))
    
    @staticmethod
    def get_balls_info():
        balls = Balls.get_balls()
        return [np.concatenate((ball['p'], ball['v'])) for ball in balls]

    # This is the distance between a point and a line segment.
    @staticmethod
    def dist_p_to_seg(P, A, B):
        AP = P - A
        AB = B - A
        t = max(0, min(1, np.dot(AP, AB) / np.dot(AB, AB)))
        return np.linalg.norm(P - (A + t * AB))

    # This makes the full Jacobian for the task space.
    # The main task space is the distance from the blade to the ball.
    # We just compute this numerically by seeing how the distance changes for
    # small changes in the end effector position.
    def JFull(self, q, x):
        ptip, Rtip, Jv, Jw = self.chain.fkin(q)
        L = self.blade_length
        J2 = np.vstack((Rtip.T @ Jv, Rtip.T @ Jw))
        T0T = T_from_Rp(Rtip, ptip)
        xt = p_from_T(np.linalg.inv(T0T) @ T_from_Rp(Reye(), x))
        dx = 0.0001
        A = np.array([0, 0, self.blade_guard_offset])
        B = np.array([0, 0, L])
        dist = self.dist_p_to_seg(xt, A, B)
        J1 = np.array([
            (self.dist_p_to_seg(xt, A + np.array([dx, 0, 0]), B + np.array([dx, 0, 0])) - dist) / dx,
            (self.dist_p_to_seg(xt, A + np.array([0, dx, 0]), B + np.array([0, dx, 0])) - dist) / dx,
            (self.dist_p_to_seg(xt, A + np.array([0, 0, dx]), B + np.array([0, 0, dx])) - dist) / dx,
            (self.dist_p_to_seg(xt, A, B + np.array([0, -dx, 0])) - dist) / dx,
            (self.dist_p_to_seg(xt, A, B + np.array([dx, 0, 0])) - dist) / dx,
            0
        ])
        return J1 @ J2

        

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        pos = self.init_balls_info[0][:3] + 5 * self.init_balls_info[0][3:]
        pos2 = self.init_balls_info[1][:3] + self.init_balls_info[1][3:]

        # We just initialy go to the first ball (distance is 0).
        if (t < 5):
            rd, rdotd = goto(t, 5, self.r0, 0)
        else:
            rd = 0
            rdotd = 0
        
        qdlast = self.qd
        Jp = self.JFull(qdlast, pos)
        ptip, Rtip, _, _ = self.chain.fkin(qdlast)
        ptip = ptip + Rtip @ np.array([0, 0, self.blade_guard_offset])
        dist_pos = self.dist_p_to_seg(pos, ptip, ptip + Rtip @ np.array([0, 0, self.blade_length]))
        dist_pos2 = self.dist_p_to_seg(pos2, ptip, ptip + Rtip @ np.array([0, 0, self.blade_length]))
        xr = rdotd + self.lam * (rd - dist_pos)

        # The primary task reduces the distance between the blade and the first ball. The secondary task reduces the distance between the blade and the second ball.
        qddot = self.inv(Jp, 1) * xr + ((np.eye(7) - self.inv(Jp, 1) @ Jp) @ self.inv(self.JFull(qdlast, pos2), 2) * -10 * dist_pos2)
        qd = qdlast + (qddot * dt)

        self.qd = qd

        pd = ptip
        Rd = Rtip
        vd = pzero()
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
    # Makes two balls and puts them randomly in the scene.
    balls.add_ball(np.array([np.random.uniform(-.5, .5), 0.5, np.random.uniform(0, .5)]), np.array([0, 0, 0]))
    balls.add_ball(np.array([np.random.uniform(-.5, .5), 0.5, np.random.uniform(0, .5)]), np.array([0, 0, 0]))
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
