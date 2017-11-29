# class for aggregation simulation environment

# hold all data and computation related to the physical environment

# in every step, it can output observation of each robot
# and take actions as input, calculate the reward, and output new observation again


from Tkinter import *

class AggrEnv():  # abbreviation for aggregation environment
    def __init__(self, robot_quantity):
        self.N = robot_quantity
        self.root = Tk()  # root window, as the simulation window
        self.root.resizable(width=False, height=False)  # inhibit resizing
        self.root.geometry('600x600+100+100')  # widthxheight+xpos+ypos
        self.root.title('Swarm Aggregation with RL of Policy Gradient')
        self.canvas = Canvas(self.root)
        self.canvas.pack(fill=BOTH)  # fill entire window
        self.dot1 = self.canvas.create_oval(10,10,30,30)
        # self.root.mainloop()
        print('program goes here')

    def frame_update(self, x, y):
        self.canvas.move(self.dot1, x, y)
        self.root.update()