"""balldemo.py

   Simulate a non-physical ball and publish as a visualization marker
   array to RVIZ.

   Node:      /balldemo
   Publish:   /visualization_marker_array   visualization_msgs.msg.MarkerArray

"""

import rclpy
import numpy as np

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

    def __init__(self, name, rate):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Prepare the publisher (latching for new subscribers).
        quality = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.pub = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)

        # Initialize the ball position, velocity, set the acceleration.
        self.radius = 0.01

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

    def add_ball(self, p, v):
        ball = {
            'p': p,
            'v': v,
            'marker': Marker()
        }
        ball['marker'].header.frame_id  = "world"
        ball['marker'].header.stamp     = self.get_clock().now().to_msg()
        ball['marker'].action           = Marker.ADD
        ball['marker'].ns               = "point"
        ball['marker'].id               = len(self.markerarray.markers) + 1
        ball['marker'].type             = Marker.SPHERE
        ball['marker'].pose.orientation = Quaternion()
        ball['marker'].pose.position    = Point_from_p(p)
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