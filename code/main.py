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
        self.node = node 
        # Set up the kinematic chain object.
        self.full_chain = KinematicChain(node, 'world', 'lightsaber_blade_end', self.jointnames())
        self.blade_rot_chain = KinematicChain(node, 'world', 'lightsaber_blade_tilt', self.jointnames()[:-1])

        self.q0 = np.array([0, 0, 0, 0, 0, 0, 0, 0.4, np.pi/2, np.pi/2, 0])
        self.p0, self.R0, _, _ = self.full_chain.fkin(self.q0)

        self.qd = self.q0.copy()
        self.lam = 20
        self.angle_lam = 1
        self.lam_perp = 20
        self.blade_len_max = 0.75
        self.blade_len_min = 0.05
        
        self.tracking = False
        self.traj_start = 0
        self.traj_time = 1.5
        self.traj_p0 = self.p0
        self.traj_pf = None

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7', 'light_saber_blade_len', 'light_saber_blade_pan', 'light_saber_blade_tilt', 'light_saber_blade_end']

    # We use weighted inverse becuase there seems to be a lot of singularities.
    @staticmethod
    def winv(J, W):
        return W @ np.linalg.pinv(J @ W)
    
    @staticmethod
    def get_balls_info():
        balls = Balls.get_balls()
        return [np.concatenate((ball['p'], ball['v'])) for ball in balls]
    

    @staticmethod
    def compute_angle(vec):
        cos_theta = np.cos(np.arcsin(vec[2]))
        # Prevent division by zero
        if np.isclose(cos_theta, 0):
            cos_theta = np.finfo(float).eps
        pan = np.arctan2(-vec[0] / cos_theta, vec[1] / cos_theta)
        tilt = np.arcsin(vec[2])
        return np.array([pan, tilt])
    
    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
        try: 
            vel = self.get_balls_info()[0][3:]
            ballpos = self.get_balls_info()[0][:3]
            pos = ballpos
            if (vel == np.zeros(3)).all():
                vel = np.array([0, -1, 0])
            dangle = self.compute_angle(vel/-np.linalg.norm(vel))
            W = np.diag([1, 1, 1, 1, 1, 1, 1, 0.5, 5, 5, 10])
            
            if(not self.tracking):
                if (self.traj_pf is None):
                    self.traj_pf = pos + vel * self.traj_time
                pos, vel = spline(t - self.traj_start, self.traj_time, self.traj_p0, self.traj_pf, pzero(), vel)
                if (t - self.traj_start >= self.traj_time):
                    self.tracking = True
            
            qdlast = self.qd
            _, Rrot, _, Jwrot = self.blade_rot_chain.fkin(qdlast[:-1])
            Jwrot = np.array([[0, 0, 1], [1, 0, 0]]) @ np.hstack((Jwrot, np.zeros((3, 1))))
            ptip, Rtip, Jvtip, _ = self.full_chain.fkin(qdlast)
            if (qdlast[7] > self.blade_len_max):
                Jvtip[:, 7] = np.zeros(3)
                qdlast[7] = self.blade_len_max
            if (qdlast[7] < self.blade_len_min):
                Jvtip[:, 7] = np.zeros(3)
                qdlast[7] = self.blade_len_min
            qddot = self.winv(Jwrot, W) @ (self.angle_lam * (dangle - self.compute_angle(Rrot[:, 2])))
            qddot += (np.eye(11) - self.winv(Jwrot, W) @ Jwrot) @ (self.winv(Jvtip, W) @ (vel + (self.lam * ep(pos, ptip))))
            qddot += (np.eye(11) - self.winv(Jwrot, W) @ Jwrot) @ (np.eye(11) - self.winv(Jvtip, W) @ Jvtip) @ (self.lam_perp * np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, np.pi/2 - qdlast[9], 0]))
            self.qd = qdlast + qddot * dt
            self.qd[7] = np.clip(self.qd[7], self.blade_len_min, self.blade_len_max)
            
            if (self.qd[-1] < 0 and np.linalg.norm(ep(ballpos, ptip)) < 0.1):
                Balls.cycle_first_ball(Balls.gen_random_posvel(10))
                self.tracking = False
                self.traj_start = t
                self.traj_p0 = ptip
                self.traj_pf = None
            
            pd = ptip
            vd = pzero()
            Rd = Rtip
            wd = pzero()
            
            # Return the desired joint and task (position/orientation) pos/vel.
            return (self.qd, qddot, pd, vd, Rd, wd)
        except Exception as e:
            self.node.get_logger().error(f"Error in evaluating trajectory: {e}")
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
    balls = Balls('balls', UPDATE_RATE)
    # Makes two balls and puts them randomly in the scene.
    balls.add_ball(Balls.gen_random_posvel(2.5))
    balls.add_ball(Balls.gen_random_posvel(5))
    balls.add_ball(Balls.gen_random_posvel(7.5))
    balls.add_ball(Balls.gen_random_posvel(10))
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
