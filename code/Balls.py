"""balldemo.py

   Simulate a non-physical ball and publish as a visualization marker
   array to RVIZ.

   Node:      /balldemo
   Publish:   /visualization_marker_array   visualization_msgs.msg.MarkerArray

"""

import rclpy
import numpy as np
import time

from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Point, Vector3, Quaternion
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

from code.utils.TransformHelpers   import *


#
#   Demo Node Class
#
class Balls(Node):
    balls = []

    @staticmethod
    def get_balls():
        return Balls.balls
    
    @staticmethod
    def gen_random_posvel(offset):
        end_pos = np.array([np.random.uniform(-0.75, 0.75), 0, np.random.uniform(0.25, 1.0)])
        vel = np.array([np.random.uniform(-0.5, 0.5), -0.3, np.random.uniform(-0.05, 0.05)])
        start_pos = end_pos - (offset * vel)
        return np.concatenate((start_pos, vel)), time.time() + offset
        
    
    @staticmethod
    # Moves the first ball to the end and then moves it to pos with vel
    def cycle_first_ball(posvel):
        ball = Balls.balls.pop(0)
        ball['p'] = posvel[0][:3]
        ball['v'] = posvel[0][3:]
        ball['spawn_time'] = posvel[1]
        Balls.balls.append(ball)
        

    def __init__(self, name, rate):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Prepare the publisher (latching for new subscribers).
        quality = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.pub = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)

        # Initialize the ball position, velocity, set the acceleration.
        self.radius = 0.05

        # Create the marker array message.
        self.markerarray = MarkerArray(markers = [])

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)

        # Create a timer to keep calling update().
        self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    def add_ball(self, pv):
        ball = {
            'p': pv[0][:3],
            'v': pv[0][3:],
            'spawn_time': pv[1],
            'marker': Marker()
        }
        ball['marker'].header.frame_id  = "world"
        ball['marker'].header.stamp     = self.get_clock().now().to_msg()
        ball['marker'].action           = Marker.ADD
        ball['marker'].ns               = "point"
        ball['marker'].id               = len(self.markerarray.markers) + 1
        ball['marker'].type             = Marker.SPHERE
        ball['marker'].pose.orientation = Quaternion()
        ball['marker'].pose.position    = Point_from_p(pv[0][:3])
        ball['marker'].scale            = Vector3(x = 2 * self.radius, y = 2 * self.radius, z = 2 * self.radius)
        ball['marker'].color            = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        self.markerarray.markers.append(ball['marker'])
        Balls.balls.append(ball)

    # Shutdown
    def shutdown(self):
        # Destroy the node, including cleaning up the timer.
        self.destroy_node()

    # Return the current time (in ROS format).
    def now(self):
        return self.start + Duration(seconds=self.t)

    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        for ball in Balls.balls:
            ball['p'] += ball['v'] * self.dt
            ball['marker'].pose.position = Point_from_p(ball['p'])
            ball['marker'].header.stamp  = self.now().to_msg()

        self.pub.publish(self.markerarray)