#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from sensor_msgs.msg import Range
from math import atan2, pi, asin, tanh, atan, sin, cos, e, ceil, floor, log, exp, acos
from std_msgs.msg import Int64, String, Float64, Float64MultiArray
import numpy as np
from numpy.linalg import norm
from nav_msgs.msg import Odometry, OccupancyGrid
import tf

map_width=1550    #real map dimensions in cm
map_height=1492
resolution=10.0   #grid tiles are 10x10 cm

class map_class():
    def __init__(self):
        rospy.init_node('mapping_node')
        self.occupancy_grid = OccupancyGrid()                                           #defining output occupancy grid
        self.occupancy_grid.info.resolution = resolution/100
        self.occupancy_grid.info.width = int(ceil(map_width/resolution))
        self.occupancy_grid.info.height = int(ceil(map_height/resolution))

        self.dim = int(ceil(map_width/resolution))*int(ceil(map_height/resolution))
        #self.flag = [0 for x in range(0,self.dim)]
        
        #self.occupancy_grid.data = [-1 for x in range(0,self.dim)]
        self.occupancy_grid.data = list( -1 * np.ones((1, self.dim)) )

        self.W = int(ceil(map_width/resolution))
        self.H = int(ceil(map_height/resolution))
        
        self.prob_matrix = -1 * np.ones((self.H, self.W))
        #self.prob_matrix2 = 0.5 * np.ones((self.H, self.W))
        
        self.yaw=0
        self.x=0
        self.y=0

        #SIMULATION PARAMETERS#
        self.Rmax = 3           #  sonar max range in m
        self.rov = 1           # visibility radius in m
        self.th3db = 0.5          # half-width of the sensor beam in radians
        self.pE = 0.4             # lower limit of the conditional probability
        self.pO = 0.6             # upper limit of the conditional probability
        self.deltark = 0.1         # parameter which designates the area in which the sensor measurement r takes the average value


        self.sonardata=[]
        self.noised_sonardata=np.zeros(16)
        self.sonar_coordinates=[ [10, 0],   [10, 5],  [10, 10],                 #relative sonar coordinates in cm
                                 [5, 10],   [0, 10],  [-5, 10],  [-10, 10],
                                 [-10, 5],  [-10, 0], [-10, -5], [-10, -10],                
                                 [-5, -10], [0, -10], [5, -10],  [10, -10],
                                 [10, -5]]
        self.sonar_thetas=[]
        for i in range(9):
            self.sonar_thetas.append(i*pi/8)                #sonar orientations in radians                              
        for i in range(7):
            self.sonar_thetas.append(-(7-i)*pi/8) 

    def sonar_callback(self, scan):     #sonar subscriber                                       
        self.sonardata=scan.data

    def odometry_callback(self, scan):          #odometry subscriber
        quaternion = (
            scan.pose.pose.orientation.x,
            scan.pose.pose.orientation.y,
            scan.pose.pose.orientation.z,
            scan.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion) #transforming quaternions to euler
        self.yaw = euler[2]
        self.x=scan.pose.pose.position.x
        self.y=scan.pose.pose.position.y

    def conditional_probability(self, i):
        ## TASK A ##
        
        # write the function which will calculate the conditional probability of all the cells affected 
        # by single sonar measurment according to the inverse sensor model
        
        r = self.sonardata[i]
        if r < 0:
            r = 0
        
        norm_sc = norm(self.sonar_coordinates[i])
        #print norm_sc, 'norma'
        theta = self.sonar_thetas[i]
        sc = [ norm_sc*cos(self.yaw + theta), norm_sc*sin(self.yaw + theta) ]
        
        xs = sc[0]
        ys = sc[1]
        
        for iy in range(0, self.H):
            
            for ix in range(0, self.W):
                
                d = [ (ix+0.5)*resolution-self.x*100-xs, (iy+0.5)*resolution-self.y*100-ys ]
                #ro = np.sqrt( d[0]**2 + d[1]**2 )/100
                ro = norm(d)/100
                
                #th = np.dot(sc, d)/( norm(d)*norm(sc) )
                #th = acos(th)
                theta_polja = atan2(d[1], d[0]) - (self.yaw + theta)
                th = atan2( sin(theta_polja),  cos(theta_polja))
                
                if (((resolution*ix)**2+(iy*resolution)**2) < norm_sc**2) or (ro > r+4*self.deltark) or (abs(th) > self.th3db):
                    continue
                
                if abs(th) <= self.th3db: #pi/16:  
                    alpha = 1 - (th/self.th3db)**2
                else:
                    alpha = 0

                dro = 1 - (1 + tanh( 2*(ro-self.rov) ))/2
                #print ro, r,  2*self.deltark
                if ro < (r - self.deltark):
                    P = 0.5 + (self.pE - 0.5)*alpha*dro
                    #print 'u rupi je!'
                    #print P
                elif (ro >= r - self.deltark) and (ro < r - self.deltark):
                    P = 0.5 + (self.pE - 0.5)*alpha*dro*( 1 - (2 + (ro-r)/self.deltark )**2 )
                    #print 'na uzbrdici'
                    #print P
                elif (ro >= r - self.deltark) and (ro < r + self.deltark):
                    P = 0.5 + (self.pO - 0.5)*alpha*dro*( 1 - ( (ro-r)/self.deltark )**2 )
                    #print 'na padini'
                    #print P
                elif (ro > r + self.deltark):
                    P = 0.5
                    #print 'u nizinama'
                    #print P
                    
                #idx = iy * self.H + ix
                #idx = ix*self.W + iy
                
                #if idx >= len(self.flag):
                #    idx = len(self.flag)-1
                
                #if self.flag[iy][ix] == 0:
                    
                if self.prob_matrix[iy][ix] == -1:
                    #self.occupancy_grid.data[idx] = P*100
                    self.prob_matrix[iy][ix] = P*100

                else:
                    #print P
                    l_odds_old = log( self.prob_matrix[iy][ix]/(100-self.prob_matrix[iy][ix]) )
                    l_odds_new = log(P/(1-P)) + l_odds_old #+ log(self.pE/self.pO)
                    #print l_odds_new

                    #try:
                        #self.occupancy_grid.data[idx] = 100/(1 + exp(-l_odds_new))
                    #print  self.prob_matrix[iy][ix]
                    self.prob_matrix[iy][ix] = 100/(1 + exp(-l_odds_new))
                    #print self.prob_matrix[iy][ix]
                    #except:
                    #    pass

                    #if self.prob_matrix[iy][ix] > 50:
                    #    self.prob_matrix2[iy][ix] = 100
                    #else:
                    #    self.prob_matrix2[iy][ix] = 0
    
                            
                    #self.flag[iy][ix] = 1

    def do_mapping(self):       
        ##TASKS B and C##

        # add the recursive cell occupancy calculation with which a new occupancy probability will be calculated for the cells which were
        # affected by the measurement i and update the values of the occupancy grid map
        
        #self.flag = [0 for x in range(0,self.dim)]
        #self.flag = np.zeros((self.H, self.W))
        
        #i = 10
        for i in range(16): #[0, 1, 2 , 14, 15]: #range(16):
            if self.sonardata[i] < self.Rmax:
                self.conditional_probability(i)
            
        self.occupancy_grid.data = self.prob_matrix.flatten().tolist()
               
    def run(self):

        self.sub= rospy.Subscriber('/robot0/sonar_data',Float64MultiArray, self.sonar_callback)       #defining the subscribers and publishers
        #self.sub= rospy.Subscriber('/robot0/odom_drift',Odometry, self.odometry_callback)
        self.sub= rospy.Subscriber('/robot0/odom',Odometry, self.odometry_callback)              #you can choose will you use real or noised odom data by commenting the subscriber code

        self.OG_publisher = rospy.Publisher('OG_map', OccupancyGrid, queue_size=10)     #occupancy grid publisher
        #self.start = 0
        r=rospy.Rate(60)
        try:
            while not rospy.is_shutdown():
                if len(self.sonardata) > 0:
                    self.do_mapping()
                    #self.occupancy_grid.data[10*self.row_width+1]=100
                    self.OG_publisher.publish(self.occupancy_grid)    #run the code and publish the occupancy grid
                    
                    r.sleep()
        except rospy.ROSInterruptException:                        
            pass                                            

if __name__ == '__main__':         
    mapping = map_class()                                                             
    try:
        mapping.run()
    except rospy.ROSInterruptException:
        pass
