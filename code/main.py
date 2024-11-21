import rclpy
import numpy as np

from math import pi, sin, cos, acos, atan2, sqrt, fmod, exp

# Grab the utilities
from code.GeneratorNode      import GeneratorNode
from code.TransformHelpers   import *
from code.TrajectoryUtils    import *

# Grab the general fkin from HW6 P1.
from code.KinematicChain     import KinematicChain


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain = KinematicChain(node, 'world', 'iiwa_link_ee', self.jointnames())

        self.q0 = np.radians(np.array([0, 90, 0, -90, 0, 0, 0]))
        self.p0 = np.array([0, 0.55, 1])
        self.R0 = Reye()
        
        self.pleft = np.array([0.3, 0.5, 0.15])
        self.Rleft = Rotx(np.radians(-90)) @ Roty(np.radians(-90))
        
        self.pright = np.array([-0.3, 0.5, 0.15])
        self.Rright = Reye()

        self.qd = self.q0
        self.lam = 20

    # Declare the joint names.
    def jointnames(self):
        # Return a list of joint names FOR THE EXPECTED URDF!
        return ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7']

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):

        if (t < 3):
            pd, vd = goto(t, 3, self.p0, self.pright)
            Rd = Reye()
            wd = pzero()
        else:
            teff = (t - 3) % 10
            s, sdot = goto(teff, 5, 0, -120) if teff < 5 else goto(teff - 5, 5, -120, 0)
            
            pd = np.array((-0.3-s*(0.6/120), 0.5, 0.55-(0.4/60**2)*(s+60)**2))
            vd = np.array((-0.6 * sdot / 120, 0, (-0.8/60**2)*(s + 60)*sdot))
            what = np.array((1, 1, -1)) / sqrt(3)
            Rd = Rotn(what, np.radians(s))
            wd = what * np.radians(sdot)

        qdlast = self.qd
        
        ptip, Rtip, Jv, Jw = self.chain.fkin(qdlast)
        xr = np.concatenate((vd, wd)) + self.lam * np.concatenate((ep(pd, ptip), eR(Rd, Rtip)))
        qddot = np.linalg.pinv(np.vstack((Jv, Jw))) @ xr
        qd = qdlast + (qddot * dt)
        
        self.qd = qd
        
        # Return the desired joint and task (position/orientation) pos/vel.
        return (qd, qddot, pd, vd, Rd, wd)


#
#  Main Code
#
def main(args=None):
    # Initialize ROS.
    rclpy.init(args=args)

    # Initialize the generator node for 100Hz udpates, using the above
    # Trajectory class.
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, meaning keep running (taking care of the timer callbacks
    # and message passing), until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
