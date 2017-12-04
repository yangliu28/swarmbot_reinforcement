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


from Tkinter import *
import numpy as np
import math


class AggrEnv():  # abbreviation for aggregation environment
    ROBOT_SIZE = 10  # diameter of robot in pixels
    ROBOT_RAD = ROBOT_SIZE/2  # radius of robot, for compensation
    ROBOT_COLOR = 'blue'
    def __init__(self, robot_quantity, world_size_physical, world_size_display,
                 sensor_range, frame_speed,
                 view_div, award_rings):
        self.N = robot_quantity
        self.size_p = world_size_physical  # side length of physical world, floating point
        self.size_d = world_size_display  # side length of display world in pixels, integer
        self.range = sensor_range  # range of communication and sensing
        self.speed = frame_speed  # physical distance per frame
        self.view_div = view_div  # how many sectors to divide the 360 view
        self.sec_wid = math.pi*2 / self.view_div  # sector width
        self.range_div = len(award_rings)  # number of award rings
        self.award_rings = award_rings  # the awards distributed for distance rings
        self.ring_wid = self.range/self.range_div  # width of the rings
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
        self.root.protocol('WM_DELETE_WINDOW', self.close_window_x)
        self.window_closed = False  # flag for window closed
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
        # create the connection map
        self.dists = []
        self.conns = []  # the connection map
        self.lines = []  # lines representing connections on canvas
        self.connections_update()

    # update the poses_d, the display position of all robots
    def poses_d_update(self):
        for i in range(self.N):
            pos_x = int(self.poses_p[i][0]/self.size_p * self.size_d)
            pos_y = int((1-self.poses_p[i][1]/self.size_p) * self.size_d)
            self.poses_d[i] = np.array([pos_x, pos_y])

    # update the distances and connection map
    def connections_update(self):
        self.dists = np.zeros((self.N, self.N))
        self.conns = np.zeros((self.N, self.N))
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                dist = np.linalg.norm(self.poses_p[i]-self.poses_p[j])
                self.dists[i,j] = dist
                self.dists[j,i] = dist
                if dist < self.range:
                    self.conns[i,j] = 1
                    self.conns[j,i] = 1

    # return the current observations of the robots, 
    def get_observations(self):
        observations = np.ones((self.N, self.view_div))
        has_neighbor = [False for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if j == i: continuous
                if self.conns[i,j] > 0:
                    # new neighbor identified, set the has_neighbor flag
                    if not has_neighbor[i]: has_neighbor[i] = True
                    # find out which sector it belongs to, and calculate distance ratio
                    vec = self.poses_p[j]-self.poses_p[i]  # vector from host to neighbor
                    ang_diff = math.atan2(vec[1], vec[0]) - self.heading[i]
                        # angle diff from heading direction to the neighbor
                    ang_diff = self.set_radian_positive(ang_diff  + self.sec_wid/2)
                        # compensate for half sector width
                    sec_index = int(ang_diff / self.sec_wid)
                    dist_ratio = self.dist[i,j] / self.range
                    if dist_ratio < observations[i, sec_index]:
                        # only the closest neighbor in that sector will be recorded
                        observations[i, sec_index] = dist_ratio
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
        # update the connection map
        self.connections_update()
        # calculate the rewards
        rewards = np.zeros((self.N))
        for i in range(self.N):
            for j in range(self.N):
                if j == i: continue
                if self.conns[i,j] > 0:
                    # accumulate the rewards
                    ring_index = int(self.dists[i,j]/self.ring_wid)
                    rewards[i] = rewards[i] + self.award_rings[ring_index]
        return rewards

    # update the display once
    def display_update(self):
        self.poses_d_update()  # re-calculate the display positions
        # for the positions of robots on canvas
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

    def close_window_x(self):
        self.window_closed = True  # reverse exit flag
        self.root.destroy()

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


