# main program for the aggregation simulation with policy gradient learning

# interact with the environment, and brain of reinforcement learning


# the graphic updating frequency is set to the same as the training data generating
# frequency, the moving step needs to be small in graphics to make it smooth
# but I did do so, because training data may need to be generated at a lower frequency
# for the good of the training, the graphics needs to sacrifice


from aggr_env import AggrEnv
import time

robot_quantity = 30
world_size_physical = 100.0  # side length of physical world
world_size_display = 600  # side length of display world, in pixels
sensor_range = 10.0  # range of communication and sensing
frame_speed = 2.0  # speed of the robot in physical world, distance per frame
view_div = 36  # divide the 360 view into how many slices

sim_env = AggrEnv(robot_quantity, world_size_physical, world_size_display,
                  sensor_range, frame_speed,
                  view_div)
sleep_time = 0.5


# more save way is checking window_closed right before updating the display

while not sim_env.window_closed:
    sim_env.display_update()
    time.sleep(sleep_time)


