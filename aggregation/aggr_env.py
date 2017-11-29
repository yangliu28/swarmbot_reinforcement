# class for aggregation simulation environment

# hold all data and computation related to the physical environment

# in every step, it can output observation of each robot
# and take actions as input, calculate the reward, and output new observation again




# window size is fixed, so pass in an appropriate robot quantity
# may enable the ability to adapting window size to robot quantity

# under Canvas class: move, coords

# 


from Tkinter import *
import random

ROBOT_SIZE = 10  # diameter of robot in pixels

class AggrEnv():  # abbreviation for aggregation environment
    def __init__(self, robot_quantity, ):
        self.N = robot_quantity
        self.world = (600, 600)  # the size of playground in pixels
        # root window, as the simulation window
        self.root = Tk()
        self.root.resizable(width=False, height=False)  # inhibit resizing
        win_size = (self.world[0]+ROBOT_SIZE, self.world[1]+ROBOT_SIZE)
        win_size_str = str(win_size)+'x'+str(win_size)
        self.root.geometry(win_size_str + '+100+100')  # widthxheight+xpos+ypos
        self.root.title('Swarm Aggregation with RL of Policy Gradient')
        # self.root.iconbitmap('../ants.png')
        ico_img = PhotoImage(file='../ants.png')  # under parent directory
        self.root.tk.call('wm', 'iconphoto', self.root._w, ico_img)  # set window icon
        # canvas for all drawing
        self.canvas = Canvas(self.root)
        self.canvas.pack(fill=BOTH)  # fill entire window
        self.dot1 = self.canvas.create_oval(10,10,30,30)
        self.canvas.coords(self.dot1)
        # self.root.mainloop()  # not needed, have my own loop
        
    # def draw

    def frame_update(self, x, y):
        self.canvas.move(self.dot1, x, y)
        self.root.update()



