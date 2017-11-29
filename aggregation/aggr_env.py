# class for aggregation simulation environment

# hold all data and computation related to the physical environment

# in every step, it can output observation of each robot
# and take actions as input, calculate the reward, and output new observation again


from Tkinter import *

class AggrEnv():  # abbreviation for aggregation environment
    def __init__(self, robot_quantity):
        self.N = robot_quantity
        # root window, as the simulation window
        self.root = Tk()
        self.root.resizable(width=False, height=False)  # inhibit resizing
        self.root.geometry('600x600+100+100')  # widthxheight+xpos+ypos
        self.root.title('Swarm Aggregation with RL of Policy Gradient')
        # self.root.iconbitmap('../ants.png')
        ico_img = PhotoImage(file='../ants.png')
        self.root.tk.call('wm', 'iconphoto', self.root._w, ico_img)
        # canvas for all drawing
        self.canvas = Canvas(self.root)
        self.canvas.pack(fill=BOTH)  # fill entire window
        self.dot1 = self.canvas.create_oval(10,10,30,30)
        # self.root.mainloop()  # not needed, have my own loop

    def frame_update(self, x, y):
        self.canvas.move(self.dot1, x, y)
        self.root.update()



