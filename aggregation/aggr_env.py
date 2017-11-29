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


# object manipulaltion under Canvas class: move, coords


from Tkinter import *
import random

ROBOT_SIZE = 10  # diameter of robot in pixels
ROBOT_COLOR = 'blue'
FRAME_SPEED = 5  # speed of the robot, defined as number of pixels per frame

class AggrEnv():  # abbreviation for aggregation environment
    def __init__(self, robot_quantity, world_size_physical, world_size_display, ):
        self.N = robot_quantity
        self.p_size = world_size_physical  # side length of physical world, floating point
        self.d_size = world_size_display  # side length of display world in pixels, integer
        self.world = world_size  # in pixels; square world
        # root window, as the simulation window
        self.root = Tk()
        self.root.resizable(width=False, height=False)  # inhibit resizing
        win_size = (self.world+ROBOT_SIZE, self.world+ROBOT_SIZE)
            # maker window larger so as to compensate the robot size
        win_size_str = str(win_size[0])+'x'+str(win_size[1])
        self.root.geometry(win_size_str + '+100+100')  # widthxheight+xpos+ypos
        self.root.title('Swarm Aggregation with RL of Policy Gradient')
        # self.root.iconbitmap('../ants.png')  # not working
        ico_img = PhotoImage(file='../ants.png')  # image file under parent directory
        self.root.tk.call('wm', 'iconphoto', self.root._w, ico_img)  # set window icon
        # canvas for all drawing
        self.canvas = Canvas(self.root)
        self.canvas.pack(fill=BOTH, expand=True)  # fill entire window
            # keyword 'fill' alone seems not working, have to add 'expand'
        # create the robots
        self.robots = []  # robots as the objects on canvas
        self.poses = []
        for _ in range(self.N):
            x0 = random.randrange(0,self.world)
            y0 = random.randrange(0,self.world)
            self.poses.append([x0,y0])
            self.robots.append(self.canvas.create_oval(
                x0,y0,x0+ROBOT_SIZE,y0+ROBOT_SIZE, outline=ROBOT_COLOR, fill=ROBOT_COLOR))
        # self.root.mainloop()  # not needed, have my own loop


    def frame_update(self):
        self.root.update()



