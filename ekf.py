#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from sensor_msgs.msg import Range
from math import atan2, pi, asin, tanh, atan, sin, cos, e, ceil, floor
from std_msgs.msg import Int64, String, Float64, Float64MultiArray
import numpy as np
from numpy.linalg import norm, inv, pinv, det
from nav_msgs.msg import Odometry, OccupancyGrid
import tf

from time import time

res = 0.1

class ekf_class():
    def __init__(self):
        rospy.init_node('ekf_node')
        self.sonar_coordinates=[ [0.10, 0.0],   [0.10, 0.05],  [0.10, 0.10],
                                 [0.05, 0.10],   [0.0, 0.10],  [-0.05, 0.10],  [-0.10, 0.10],
                                 [-0.10, 0.05],  [-0.10, 0.0], [-0.10, -0.05], [-0.10, -0.10],
                                 [-0.05, -0.10], [0.0, -0.10], [0.05, -0.10],  [0.10, -0.10],
                                 [0.10, -0.05]]
        self.sonar_thetas=[]
        for i in range(9):
            self.sonar_thetas.append(i*pi/8)
        for i in range(7):
            self.sonar_thetas.append(-(7-i)*pi/8)
        self.sonardata=[]
        self.yaw=0
        self.x=0
        self.y=0
        self.wheel_velocity=Twist()

        self.odom_corr = Odometry()
        self.odom_corr.child_frame_id = "robot0"
        self.odom_corr.header.frame_id = "map_static"

        self.prepreke = []

        self.x_corr = 0
        self.y_corr = 0
        self.yaw_corr = 0

        self.P = 1000 * np.eye(3)

        self.f = 30
        self.start = 0

        self.dTh = 0
        self.D = 0

        self.T = 1./self.f
        self.pocni = 0

        self.tb = tf.TransformBroadcaster()

    def sonar_callback(self, scan):     #sonar subscriber
        self.sonardata=scan.data

    def odometry_callback(self, scan):          #odometry subscriber

        quaternion = (
            scan.pose.pose.orientation.x,
            scan.pose.pose.orientation.y,
            scan.pose.pose.orientation.z,
            scan.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion) #transforming quaternions to euler
        #self.roll = euler[0]
        #self.pitch = euler[1]
        self.yaw = euler[2]
        self.x=scan.pose.pose.position.x
        self.y=scan.pose.pose.position.y

        self.wheel_velocity = scan.twist.twist
        self.pocni = 1

    def wrap_pi(self, phi):

        #return phi - int( round(phi/(2*pi)) ) * 2 * pi
        return atan2(sin(phi), cos(phi))

    def sonarpredict(self, x, y, yaw):

        H = np.empty(0)
        h_m = np.empty(0)
        y_out = np.empty(0)

        min_dps = -1 * np.ones(16)
        p_mins = np.zeros((16, 2))

        for p in self.prepreke:

            dist = norm(np.array([p[0] - x, p[1] - y]))

            if dist < 3.25:

                for i in range(16):

                    # print 'sonar', i

                    norm_sc = norm(self.sonar_coordinates[i])
                    # print norm_sc, self.sonardata[i]
                    angle_sc = atan2(self.sonar_coordinates[i][1], self.sonar_coordinates[i][0])
                    # print angle_sc, self.sonar_thetas[i], yaw
                    #r = self.sonardata[i]

                    xs = x + norm_sc * cos(yaw + angle_sc)
                    ys = y + norm_sc * sin(yaw + angle_sc)

                    # print x-xs, y-ys

                    # min_dp = -1
                    p_min = [0, 0]
                    # for p in self.prepreke:
                    dp = np.array([p[0] - xs, p[1] - ys])

                    th1 = atan2(dp[1], dp[0]) - yaw - self.sonar_thetas[i]
                    angle = atan2(sin(th1), cos(th1))

                    if abs(angle) >= 0.87 / 2:
                        continue

                    # print angle, 'angle'

                    adp = norm(dp)
                    if adp < min_dps[i] or min_dps[i] == -1:  # adp < min_dp or min_dp == -1:
                        min_dps[i] = adp
                        p_mins[i] = p
                        # psi_min = angle

                    # print min_dp, r

        for i in range(16):
            if min_dps[i] < 3 and abs(min_dps[i] - self.sonardata[i]) < 8*res:
                # print psi_min

                xp = p_mins[i][0]  # xs + min_dp * cos(psi_min + yaw + self.sonar_thetas[i])
                yp = p_mins[i][1]  # ys + min_dp * sin(psi_min + yaw + self.sonar_thetas[i])

                #print xp, yp

                #print min_dps[i] - self.sonardata[i], 'diff'
                H = np.append(H, list(1 / np.sqrt((x - xp) ** 2 + (y - yp) ** 2) * np.array([x - xp, y - yp, 0])))
                h_m = np.append(h_m, min_dps[i])
                y_out = np.append(y_out, self.sonardata[i])

        H = np.reshape(H, (len(H) / 3, 3))
        h_m = h_m.reshape(-1, 1)
        y_out = y_out.reshape(-1, 1)
        V = np.eye(H.shape[0])
        #print H
        #print H.shape

        return H, V, y_out, h_m

    def correct(self):

        dTh = self.wheel_velocity.angular.z * self.T
        D = self.wheel_velocity.linear.x * self.T

        Q = np.array([[10 * ((dTh*pi/180) ** 2), 0], [0, 10 ** ( -4)]])

        if self.start == 0:
            self.yaw_corr = self.yaw
            self.x_corr = self.x
            self.y_corr = self.y

            print self.yaw_corr, self.x_corr, self.y_corr, 'start'

            self.start = 1

        yaw_corr = self.wrap_pi(self.yaw_corr)

        A = np.array([[1, 0, -D * sin(yaw_corr + dTh)],
                      [0, 1, D * cos(yaw_corr + dTh)],
                      [0, 0, 1]])

        W = np.array([[-D * sin(yaw_corr + dTh), cos(yaw_corr + dTh)],
                      [D * cos(yaw_corr + dTh), sin(yaw_corr + dTh)],
                      [1, 0]])

        P_m = np.dot(np.dot(A, self.P), A.T) + np.dot(np.dot(W, Q), W.T)

        #x_m = self.model_function(self.x_corr, self.y_corr, yaw_corr, D, dTh)

        x_m = np.array([[self.x_corr + D*cos(yaw_corr+dTh)],
                        [self.y_corr + D*sin(yaw_corr+dTh)],
                        [yaw_corr + dTh]])

        print(x_m)

        H, V, y_out, h_m = self.sonarpredict(x_m[0][0], x_m[1][0], x_m[2][0])
        R = 0.01 * np.eye(H.shape[0])

        S = np.dot(np.dot(H, P_m), H.T) + np.dot(np.dot(V, R), V.T)

        if H.shape[0] != 0:
            if det(S) != 0:

                K = np.dot(np.dot(P_m, H.T), inv(S))
                self.P = P_m - np.dot(np.dot(K, S), K.T)
                #print 'razlika izlaza', y_out-h_m
                x_p = x_m + np.dot(K, y_out - h_m)

            else:
                x_p = x_m
        else:
            x_p = np.array([[self.x_corr],
                            [self.y_corr],
                            [self.yaw_corr]])

        self.odom_corr.twist.twist.linear.x = np.sqrt((x_p[0][0] - self.x_corr) ** 2 + (x_p[1][0] - self.y_corr) ** 2) / self.T
        self.odom_corr.twist.twist.angular.z = (self.wrap_pi(x_p[2][0]) - self.yaw_corr) / self.T

        print x_p, 'x plus'

        self.x_corr = x_p[0][0]
        self.y_corr = x_p[1][0]
        self.yaw_corr = self.wrap_pi(x_p[2][0])

    #def load_map(self):
    #    self.try_map = np.load('/home/angie/catkin_ws/src/mobile-robotics/src/resources/example_map.npy')
    #    nesto = list(self.try_map.flatten())
    #    for i, x in enumerate(nesto):

    #        if x > 0:
    #            nesto[i] = 100
    #        else:
    #            nesto[i] = 0

    #    self.mapa = np.reshape(nesto, (150, 150))

    def load_map(self):
        #self.try_map = np.load('/home/angie/catkin_ws/src/mobile-robotics/src/resources/example_map.npy')
        #self.try_map = np.load('/home/angie/catkin_ws/src/mobile-robotics/src/resources/pokusaj.npy')

        #print self.try_map

        self.try_map = np.loadtxt('/home/angie/catkin_ws/src/mobile-robotics/src/resources/map.txt', dtype='str')

        self.try_map = list(reversed(self.try_map))

        nesto = []
        for x in self.try_map:
            nesto.append([int(i) for i in x])

        nesto = np.array(nesto)
        nesto = nesto[::5, ::5]
        #print(nesto.shape)

        np.save('pokusaj.npy', nesto)

        nesto = list(nesto.flatten())

        #nesto = list(self.try_map.flatten())

        for i, y in enumerate(nesto):

            if y == 0:
                nesto[i] = 100
            elif y == 1:
                nesto[i] = 0

        #self.map.data = nesto
        #self.map_pub.publish(self.map)

        self.mapa = np.reshape(nesto, (150, 155))

        for i in range(150):
            for j in range(155):
                if self.mapa[i][j] == 100:
                    self.prepreke.append([(j+0.5)*res, (i+0.5)*res])

        #print(len(self.prepreke))
        self.prepreke = self.prepreke

    def run(self):
        rospy.Subscriber('/robot0/sonar_data',Float64MultiArray, self.sonar_callback)       #defining the subscribers
        rospy.Subscriber('/robot0/odom_drift',Odometry, self.odometry_callback)

        self.odom_corr_pub = rospy.Publisher('/robot0/odom_corr',Odometry, queue_size=10)

        #self.load_map()


        #self.mapa = np.load('/home/angie/catkin_ws/src/mobile-robotics/src/resources/example_map.npy')/2.56
        self.mapa = np.load('/home/angie/catkin_ws/src/mobile-robotics/src/resources/moja_mapica.npy')
        self.mapa[self.mapa < 0] = 50
        self.mapa[self.mapa >= 50] = 100
        self.mapa[self.mapa < 50] = 0
        #self.mapa = self.mapa.T
        print(self.mapa.shape)

        for i in range(self.mapa.shape[1]):
            for j in range(self.mapa.shape[0]):
                if self.mapa[j][i] == 100:
                    self.prepreke.append([(i+0.5)*res, (j+0.5)*res])

        self.tb.sendTransform((1, 2, 0),
                              tf.transformations.quaternion_from_euler(0, 0, 0),
                              rospy.Time.now(),
                              "robot0",  # child
                              "map_static"  # parent
                              )

        r=rospy.Rate(self.f)
        try:
            while not rospy.is_shutdown():

                if self.pocni:
                    uslo = time()

                    self.correct()

                    self.odom_corr.pose.pose.position.x = self.x_corr
                    #print self.x_corr, self.odom_corr.pose.pose.position.x, self.x
                    self.odom_corr.pose.pose.position.y = self.y_corr
                    self.odom_corr.pose.pose.position.z = 0

                    quaternion = tf.transformations.quaternion_from_euler(0, 0, self.yaw_corr)

                    self.odom_corr.pose.pose.orientation.x = quaternion[0]
                    self.odom_corr.pose.pose.orientation.y = quaternion[1]
                    self.odom_corr.pose.pose.orientation.z = quaternion[2]
                    self.odom_corr.pose.pose.orientation.w = quaternion[3]

                    self.odom_corr_pub.publish(self.odom_corr)

                    #print time() - uslo, 'vrijeme obrade'
                    self.T = 1./self.f + time() - uslo
                    #print(1/self.T)

                r.sleep()
        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    ekf = ekf_class()
    try:
        ekf.run()
    except rospy.ROSInterruptException:
        pass
