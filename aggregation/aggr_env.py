# class for aggregation simulation environment

# in every step, it can output observation of each robot
# and take actions as input, calculate the reward, and output new observation again


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

# object manipulaltion under Canvas class: move, coords


from Tkinter import *
import random

ROBOT_SIZE = 10  # diameter of robot in pixels
ROBOT_COLOR = 'blue'


class AggrEnv():  # abbreviation for aggregation environment
    def __init__(self, robot_quantity, world_size_physical, world_size_display,
                 sensor_range, frame_speed):
        self.N = robot_quantity
        self.size_p = world_size_physical  # side length of physical world, floating point
        self.size_d = world_size_display  # side length of display world in pixels, integer
        self.range = sensor_range  # range of communication and sensing
        self.speed = frame_speed  # physical distance per frame
        # root window, as the simulation window
        self.root = Tk()
        self.root.resizable(width=False, height=False)  # inhibit resizing
        win_size = (self.size_d+ROBOT_SIZE, self.size_d+ROBOT_SIZE)
            # maker window larger so as to compensate the robot size
        win_size_str = str(win_size[0])+'x'+str(win_size[1])
        self.root.geometry(win_size_str + '+100+100')  # widthxheight+xpos+ypos
        self.root.title('Swarm Aggregation with RL of Policy Gradient')
        # self.root.iconbitmap('../ants.png')  # not working
        ico_img = PhotoImage(file='../ants.png')  # image file under parent directory
        self.root.tk.call('wm', 'iconphoto', self.root._w, ico_img)  # set window icon
        # canvas for all drawing
        self.canvas = Canvas(self.root, background='white')
        self.canvas.pack(fill=BOTH, expand=True)  # fill entire window
            # keyword 'fill' alone seems not working, have to add 'expand'
        # create the robots
        self.robots = []  # robots as the objects on canvas
        self.poses_p = [0.0 for i in range(self.N)]  # robot positions in physical
        self.poses_d = [0 for i in range(self.N)]  # robot positions in display
        self.poses_d_last = self.poses_d[:]  # display positions of last frame
        for i in range(self.N):
            self.poses_p[i] = [random.uniform(0, self.size_p),
                               random.uniform(0, self.size_p)]
        self.display_pos_update()  # update the display positions of robots
        self.poses_d_last = self.poses_d[:]  # initialize poses_d_last
        for i in range(self.N):
            self.robots.append(self.canvas.create_oval(
                self.poses_d[i][0], self.poses_d[i][1],
                self.poses_d[i][0]+ROBOT_SIZE, self.poses_d[i][1]+ROBOT_SIZE,
                outline=ROBOT_COLOR, fill=ROBOT_COLOR))
        # create the connections
        self.connections = 
        # self.root.mainloop()  # not needed, have my own loop

    # update all display positions
    def display_pos_update(self):
        for i in range(self.N):
            pos_x = int(self.poses_p[i][0]/self.size_p * self.size_d)
            pos_y = int((1-self.poses_p[i][1]/self.size_p) * self.size_d)
            self.poses_d[i] = [pos_x, pos_y]

    # update the connection map
    def connection_update(self):

    def frame_update(self):
        self.root.update()




