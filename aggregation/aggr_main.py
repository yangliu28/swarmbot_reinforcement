# main program for the aggregation simulation with policy gradient learning

# interact with the environment, and brain of reinforcement learning


# the graphic updating frequency is set to the same as the training data generating
# frequency, the moving step needs to be small in graphics to make it smooth
# but I did do so, because training data may need to be generated at a lower frequency
# for the good of the training, the graphics needs to sacrifice

# (following exceptions will be performed in the main program)
# The rewards are given based on how long it can maintain certain distances to neighbors.
# Exception 1:
    # If an action leads the robot from zero neighbor to non-zero neighbors. Although the
    # result is having neighbors around and the action earns reward according to the rules,
    # and this is a valid entry in the training data, however the result is purely by luck
    # (not the result of the neural network), and should get no credit for it, so no reward
    # and it won't go into the training data set.
# Exception 2:
    # If an action leads the robot from non-zero neighbors to zero neighbor. Although the
    # result is having no neighbor around, the action gets no reward and it won't go into
    # the training data, however the neural network is the cause of the result, so the
    # action and the zero reward should be part of the training data.


from aggr_env import AggrEnv
import time

robot_quantity = 30
world_size_physical = 100.0  # side length of physical world
world_size_display = 600  # side length of display world, in pixels
sensor_range = 10.0  # range of communication and sensing
frame_speed = 2.0  # speed of the robot in physical world, distance per frame
view_div = 36  # divide the 360 view into how many slices
award_rings = (1,3,5,3,1)  # awards distributed for nested rings in the range
    # from closest to farthest

sim_env = AggrEnv(robot_quantity, world_size_physical, world_size_display,
                  sensor_range, frame_speed,
                  view_div, award_rings)
sleep_time = 0.5


# more save way is checking window_closed right before updating the display

while not sim_env.window_closed:
    sim_env.display_update()
    time.sleep(sleep_time)


