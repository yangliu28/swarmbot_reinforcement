# class for the aggregation simulation environment

# In a simulation steep, it can output the observations of the robots; it can take
# actions as input, update the physical environment, and return the rewards.


# Since tkinter draw circle on canvas by the pixel position of top left corner and right
# bottom corner, it's more natural just use the top left corner as the coordinates of the
# objects on canvas.


# Window size is fixed, so pass in an appropriate robot quantity.
# May add a feature to dynamically adjust window size, and output current world size
# so the world size can be adjusted accurately.

# There were two sets of coordinates in my previous simulation. One is the continuous
# physical world, for computing the accurate position of the robots. The other is the
# discrete pixel world, the posiitons are converted from the physical world.
# This method, although a bit complicated, make the position of robots accurate.

# Physical world coordinates and display world coordinates:
# Origin of physical world is at bottom left corner, x points to right, y to top;
# Origin of display world is at top left corner, x points to right, y to bottom.

# relation between the equally divided sectors and heading direction:
# the heading direction is in the middle of a sector.
# use [-math.pi, math.pi) as the range of heading direction

# A few ways I have tried to add the pause function:
# 1.Multi-thread the mainloop() function so can bind key pressed action to a callback. The
# correct way is build the simulation environment class as inherited from  threading.Thread.
# Following is the link to do so. But problem not only appears when "different apartment"
# error occurs, but other Tk window related operations will be on different theread, and
# incurs errors too.
# https://stackoverflow.com/questions/459083/how-do-you-run-your-own-code-alongside-tkinters-event-loop
# 2.The after() method from Tk sound a quick fix, but there isn't an appropriate way to
# check keyboard input like the callback functions.
# 3.This is the way I end up with. I directed the toplevel protocol 'WM_TAKE_FOCUS' to
# reverse the status of a variable, which indicates whether the simulation should be paused.
# (another source: https://gordonlesti.com/use-tkinter-without-mainloop/)
# In summary, to use the pause, click on any other window to loose the focus, and click the
# simulation window again to reverse the pause status.

# change the reward from pure score to the change of score
# in this case, should not use the award accumulation to trigger training
# but number of valid data entries

# Translate the observation to data format
# For each sector, if there is a neighbor, it will be represented as distance ratio. That
# is, the distance to the host robot divided by the sensing range. If there are multiple
# robots at same sector, only the closest one will be recorded. If there is no neighbor in
# a sector, it is a one, like there is a neighbor just outside the sensing range. But on
# second thought, zero might be better, because one means excitation of the activation
# function.
# A compromise for these two is to translate the observation to the degree of closeness on
# the sectors. If there is neighbor on a sector, it is represented as one subtracts the
# distance ratio of the closest robot. If there is no neighbor, it is just zero.

# Intersecting line segments of connections should be avoided when updating connections.
# the algorithm to check if two line segments intersect:
# http://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

from Tkinter import *
import numpy as np
import math
import operator

class AggrEnv():  # abbreviation for aggregation environment
    ROBOT_SIZE = 10  # diameter of robot in pixels
    ROBOT_RAD = ROBOT_SIZE/2  # radius of robot, for compensation
    ROBOT_COLOR = 'blue'
    def __init__(self, robot_quantity, world_size_physical, world_size_display,
                 sensor_range, frame_speed,
                 view_div, score_rings,
                 need_pause):
        self.N = robot_quantity
        self.size_p = world_size_physical  # side length of physical world, floating point
        self.size_d = world_size_display  # side length of display world in pixels, integer
        self.range = sensor_range  # range of communication and sensing
        self.speed = frame_speed  # physical distance per frame
        self.view_div = view_div  # how many sectors to divide the 360 view
        self.sec_wid = math.pi*2 / self.view_div  # sector width
        self.range_div = len(score_rings)  # number of award rings
        self.score_rings = score_rings  # the awards distributed for distance rings
        self.ring_wid = self.range/self.range_div  # width of the rings
        self.need_pause = need_pause  # will optionally add pause function
        # root window, as the simulation window
        self.root = Tk()
        self.root.resizable(width=False, height=False)  # inhibit resizing
        win_size = (self.size_d+self.ROBOT_SIZE, self.size_d+self.ROBOT_SIZE)
            # maker window larger so as to compensate the robot size
        win_size_str = str(win_size[0])+'x'+str(win_size[1])
        self.root.geometry(win_size_str + '+100+100')  # widthxheight+xpos+ypos
        self.root.title('Swarm Aggregation with RL of Policy Gradient')
        # self.root.iconbitmap('../ants.png')  # not working
        ico_img = PhotoImage(file='../ants.png')  # image file under parent directory
        self.root.tk.call('wm', 'iconphoto', self.root._w, ico_img)  # set window icon
        self.root.protocol('WM_DELETE_WINDOW', self.close_window)
        self.window_closed = False  # flag for window closed
        if self.need_pause:
            self.root.protocol('WM_TAKE_FOCUS', self.pause_reverse)
            self.pause_on = False
        # create the canvas, as a holder for all graphics objects
        self.canvas = Canvas(self.root, background='white')
        self.canvas.pack(fill=BOTH, expand=True)  # fill entire window
            # keyword 'fill' alone seems not working, have to add 'expand'
        # create the robots
        self.robots = []  # robots as the objects on canvas
        self.poses_p = np.random.uniform(0.0, self.size_p, (self.N,2))
            # robot positions in physical world
        self.poses_d = np.zeros((self.N, 2))  # current robot positions in display
        self.poses_d_update()  # update the poses_d from poses_p
        self.poses_d_last = np.copy(self.poses_d)  # display positions of last frame
        for i in range(self.N):
            self.robots.append(self.canvas.create_oval(
                self.poses_d[i][0], self.poses_d[i][1],
                self.poses_d[i][0]+self.ROBOT_SIZE, self.poses_d[i][1]+self.ROBOT_SIZE,
                outline=self.ROBOT_COLOR, fill=self.ROBOT_COLOR))
        self.heading = np.random.uniform(-math.pi, math.pi, (self.N))  # heading direction
        # create the connection map and score map
        self.dists = np.zeros((self.N, self.N))
        self.conns = np.zeros((self.N, self.N))  # connection map
        self.lines = []  # lines representing connections on canvas
        self.connections_init()
        self.conns_last = np.copy(self.conns)  # connection map of last state
        self.scores = np.zeros((self.N, self.N))  # score map, for calculating the rewards
        self.scores_update()  # update the variable self.scores
        self.scores_last = np.copy(self.scores)  # score map of last state

    # update the poses_d, the display position of all robots
    def poses_d_update(self):
        for i in range(self.N):
            pos_x = int(self.poses_p[i][0]/self.size_p * self.size_d)
            pos_y = int((1-self.poses_p[i][1]/self.size_p) * self.size_d)
            self.poses_d[i] = np.array([pos_x, pos_y])

    # update the distance map(self.dists), only called when updating connections
    def distances_update(self):
        self.dists = np.zeros((self.N, self.N))
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                dist = np.linalg.norm(self.poses_p[i]-self.poses_p[j])
                self.dists[i,j] = dist
                self.dists[j,i] = dist

    # initialize the connection map
    def connections_init(self):
        self.distances_update()
        new_pool = []  # pool for new connections to be considered
        approved_pool = []  # pool for all connections approved
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                if self.dists[i,j] < self.range:
                    # append distance and robot indices as tuple
                    new_pool.append((self.dists[i,j], (i,j)))
        new_pool = sorted(new_pool, key=operator.itemgetter(0))  # sort by distance
        print(new_pool)
        # check every connection from closest to farthest for intersecting
        intersecting_count = 0
        for i in range(len(new_pool)):
            conn = new_pool[i][1]  # the connection in consideration
            p00 = conn[0]  # index of robot for first line
            p01 = conn[1]
            intersecting = False
            for j in range(len(approved_pool)):
                conn_comp = approved_pool[j]  # the connection to be compared with
                p10 = conn_comp[0]
                p11 = conn_comp[1]
                if p00 == p10 or p00 == p11 or p01 == p10 or p01 == p11:
                    continue  # skip when two line segments sharing end point
                intersecting = self.check_intersect(p00, p01, p10, p11)
                if intersecting: break
            if not intersecting:
                approved_pool.append(conn)
            else:
                intersecting_count = intersecting_count + 1
        # update result to self.conns
        self.conns = np.zeros((self.N, self.N))
        for conn in approved_pool:
            self.conns[conn[0], conn[1]] = 1
            self.conns[conn[1], conn[0]] = 1
        print('%i intersecting found when initializing connections' % intersecting_count)

    # update the connection map
    def connections_update(self):
        self.distances_update()
        new_pool = []  # pool for new connections to be considered
        old_pool = []  # pool for maintained old connections
        approved_pool = []  # pool for all connections approved
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                if self.dists[i,j] < self.range:
                    if self.conns_last[i,j] > 0:
                        # this connection has been maintained
                        old_pool.append((self.dists[i,j], (i,j)))
                    else:
                        # this connection is new
                        new_pool.append((self.dists[i,j], (i,j)))
        # check old connections from closest to farthese for intersecting
        old_pool = sorted(old_pool, key=operator.itemgetter(0))
        for i in range(len(old_pool)):
            conn = old_pool[i][1]
            p00 = conn[0]
            p01 = conn[1]
            intersecting = False
            for j in range(len(approved_pool)):
                conn_comp = approved_pool[j]
                p10 = conn_comp[0]
                p11 = conn_comp[1]
                if p00 == p10 or p00 == p11 or p01 == p10 or p01 == p11: continue
                intersecting = self.check_intersect(p00, p01, p10, p11)
                if intersecting: break
            if not intersecting:
                approved_pool.append(conn)
        # check every new connection from closest to farthest for intersecting
        new_pool = sorted(new_pool, key=operator.itemgetter(0))
        intersecting_count = 0
        for i in range(len(new_pool)):
            conn = new_pool[i][1]  # the connection in consideration
            p00 = conn[0]  # index of robot for first line
            p01 = conn[1]
            intersecting = False
            for j in range(len(approved_pool)):
                conn_comp = approved_pool[j]  # the connection to be compared with
                p10 = conn_comp[0]
                p11 = conn_comp[1]
                if p00 == p10 or p00 == p11 or p01 == p10 or p01 == p11: continue
                intersecting = self.check_intersect(p00, p01, p10, p11)
                if intersecting: break
            if not intersecting:
                approved_pool.append(conn)
        # update result to self.conns
        self.conns = np.zeros((self.N, self.N))
        for conn in approved_pool:
            self.conns[conn[0], conn[1]] = 1
            self.conns[conn[1], conn[0]] = 1

    # check if line segment (p00, p01) intersects with (p10, p11)
    def check_intersect(self, p00, p01, p10, p11):
        o1 = self.orientation(p00, p01, p10)
        o2 = self.orientation(p00, p01, p11)
        o3 = self.orientation(p10, p11, p00)
        o4 = self.orientation(p10, p11, p01)
        if ((o1 != o2 and o3 != o4) or
            (o1 == 0 and self.on_segment(p00, p10, p01)) or
            (o2 == 0 and self.on_segment(p00, p11, p01)) or
            (o3 == 0 and self.on_segment(p10, p00, p11)) or
            (o4 == 0 and self.on_segment(p10, p01, p11))):
            # the two line segments intersect!
            return True
        else:
            return False

    # find orientation of ordered triplet (p, q, r)
    # return
        # 0, if p, q, r are colinear
        # 1, if clockwise
        # 2, if counterclockwise
    def orientation(self, p, q, r):
        val = ((self.poses_p[q,1] - self.poses_p[p,1]) *
               (self.poses_p[r,0] - self.poses_p[q,0]) - 
               (self.poses_p[q,0] - self.poses_p[p,0]) *
               (self.poses_p[r,1] - self.poses_p[q,1]))
        if val > 0: return 1
        elif val < 0: return 2
        else: return 0

    # given three colinear points p, q, r
    # the function checks if point q lies on line segment 'pr'
    def on_segment(self, p, q, r):
        if (self.poses_p[q,0] <= max(self.poses_p[p,0], self.poses_p[r,0]) and
            self.poses_p[q,0] >= min(self.poses_p[p,0], self.poses_p[r,0]) and
            self.poses_p[q,1] <= max(self.poses_p[p,1], self.poses_p[r,1]) and
            self.poses_p[q,1] >= min(self.poses_p[p,1], self.poses_p[r,1])):
            return True
        return False

    # update the score for each pair of robots based on the score rings
    # should be performed after connections_update()
    def scores_update(self, ):
        self.scores = np.zeros((self.N, self.N))
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                if self.conns[i,j] > 0:
                    ring_index = int(self.dists[i,j]/self.ring_wid)
                    self.scores[i,j] = self.score_rings[ring_index]
                    self.scores[j,i] = self.score_rings[ring_index]
                else:
                    self.scores[i,j] = 0
                    self.scores[j,i] = 0

    # return the current observations of the robots, 
    def get_observations(self):
        observations = np.zeros((self.N, self.view_div))
        has_neighbor = [False for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if j == i: continue
                if self.conns[i,j] > 0:
                    # new neighbor identified, set the has_neighbor flag
                    if not has_neighbor[i]: has_neighbor[i] = True
                    # find out which sector it belongs to, and calculate distance ratio
                    vec = self.poses_p[j]-self.poses_p[i]  # vector from host to neighbor
                    ang_diff = math.atan2(vec[1], vec[0]) - self.heading[i]
                        # angle diff from heading direction to the neighbor
                    ang_diff = self.set_radian_positive(ang_diff  + self.sec_wid/2)
                        # compensate for half sector width
                    sect_index = int(ang_diff / self.sec_wid)
                    dist_ratio = self.dists[i,j] / self.range
                    closeness = 1 - dist_ratio  # the degree of closeness
                    if closeness > observations[i,sect_index]:
                        # only the closest neighbor in the sector will be recorded
                        observations[i,sect_index] = closeness
        return observations, has_neighbor

    # step update (graphics operations are not included)
    # take actions as input, update the physical environment, return the rewards
    def step_update_without_display(self, actions):
        # update the new headings and physical positions
        for i in range(self.N):
            # the new heading directions
            self.heading[i] = self.heading[i] + self.sec_wid * actions[i]
            self.heading[i] = self.reset_radian(self.heading[i])
            # one step of physical positions
            head_vec = np.array([math.cos(self.heading[i]), math.sin(self.heading[i])])
            self.poses_p[i] = self.poses_p[i] + self.speed * head_vec
            # wall bouncing algorithm, keep robots inside the window
            # It is borrowed from my previous simulations in 'swarm_formation_sim', but is
            # different that positions of the robots have been bounced right away, instead
            # of using one more step of update to make it take effect.
            if self.poses_p[i,0] >= self.size_p:  # out of right boundary
                if head_vec[0] > 0:  # moving direction on x is pointing right
                    self.heading[i] = self.reset_radian(2*(math.pi/2) - self.heading[i])
                    self.poses_p[i,0] = 2*self.size_p - self.poses_p[i,0]
            elif self.poses_p[i,0] <= 0:  # out of left boundary
                if head_vec[0] < 0:  # moving direction on x is pointing left
                    self.heading[i] = self.reset_radian(2*(math.pi/2) - self.heading[i])
                    self.poses_p[i,0] = -self.poses_p[i,0]
            if self.poses_p[i,1] >= self.size_p:  # out of top boundary
                if head_vec[1] > 0:  # moving direction on y is pointing up
                    self.heading[i] = self.reset_radian(2*(0) - self.heading[i])
                    self.poses_p[i,1] = 2*self.size_p - self.poses_p[i,1]
            elif self.poses_p[i,1] <= 0:  # out of bottom boundary
                if head_vec[1] < 0:  # moving direction on y is pointing down
                    self.heading[i] = self.reset_radian(2*(0) - self.heading[i])
                    self.poses_p[i,1] = -self.poses_p[i,1]
        # calculate the rewards by the changes of the scores
        self.connections_update()  # update the connection map
        self.scores_update()  # update the score map
        rewards = np.zeros((self.N))
        for i in range(self.N):
            for j in range(self.N):
                if j == i: continue
                if self.conns_last[i,j] > 0:
                    # rewards is only relevant to maintaing old connections
                    rewards[i] = rewards[i] + (self.scores[i,j] - self.scores_last[i,j])
        self.conns_last = np.copy(self.conns)
        self.scores_last = np.copy(self.scores)
        return rewards

    # update the display once
    def display_update(self):
        self.poses_d_update()  # re-calculate the display positions
        # for the robots on canvas
        for i in range(self.N):
            move = self.poses_d[i]-self.poses_d_last[i]
            self.canvas.move(self.robots[i], move[0], move[1])
        self.poses_d_last = np.copy(self.poses_d)  # reset pos of last frame
        # for the connecting lines on canvas
        for line in self.lines:
            self.canvas.delete(line)  # erase the lines on canvas
        self.lines = []  # prepare to re-create the lines
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                if self.conns[i,j] > 0:
                    (x0, y0) = (self.poses_d[i][0] + self.ROBOT_RAD,
                                self.poses_d[i][1] + self.ROBOT_RAD)
                    (x1, y1) = (self.poses_d[j][0] + self.ROBOT_RAD,
                                self.poses_d[j][1] + self.ROBOT_RAD)
                    self.lines.append(self.canvas.create_line(
                        x0, y0, x1, y1, fill=self.ROBOT_COLOR))
        # update the new frame
        self.root.update()

    # close window routine
    def close_window(self):
        self.window_closed = True  # reverse exit flag
        self.root.destroy()

    # callback to reverse the pause flag
    def pause_reverse(self):
        self.pause_on = not self.pause_on

    # set radian to the positive range of [0, math.pi*2)
    def set_radian_positive(self, radian):
        while radian < 0:
            radian = radian + math.pi*2
        while radian >= math.pi*2:
            radian = radian - math.pi*2
        return radian

    # reset radian to the range of [-math.pi, math.pi)
    def reset_radian(self, radian):
        while radian < -math.pi:
            radian = radian + math.pi*2
        while radian >= math.pi:
            radian = radian - math.pi*2
        return radian


