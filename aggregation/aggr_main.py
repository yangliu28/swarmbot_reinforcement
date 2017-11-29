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

sim_env = AggrEnv(robot_quantity, world_size_physical, world_size_display)
print(sim_env.poses)

sleep_time = 0.1

while True:
    sim_env.frame_update()
    time.sleep(sleep_time)


