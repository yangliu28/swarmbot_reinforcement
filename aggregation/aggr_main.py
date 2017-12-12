# main program for the aggregation simulation with policy gradient

# the graphics updating frequency is set to the same as the training data generating
# frequency, the moving step needs to be small in graphics to make it smooth
# but I did not do so, because training data may need to be generated at a lower frequency
# for the good of the training, so the smooth graphics is sacrified here

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
# The above can be summarized much succinctly as, if the robot has non-zero neighbors before
# the action, the action and reward should be in the training data.

from aggr_env import AggrEnv
from aggr_pg import PolicyGradient
import time
import numpy as np

# for simulation environment
robot_quantity = 30
world_size_physical = 100.0  # side length of physical world
world_size_display = 600  # side length of display world, in pixels
sensor_range = 10.0  # range of communication and sensing, radius of the sensing circle
frame_speed = 0.5  # speed of the robot in physical world, distance per frame
view_div = 36  # divide the 360 view into how many slices
score_rings = (2,4,6,4,2)  # scores distributed for nested rings in the range
    # from closest to farthest
need_pause = True
# for policy gradient
learning_rate = 0.5
training_repeats = 100  # repeat training each episode for these times

# instantiate the aggregation environment
aggr_env = AggrEnv(robot_quantity, world_size_physical, world_size_display,
                  sensor_range, frame_speed,
                  view_div, score_rings,
                  need_pause)
# instantiate the policy gradient
PG = PolicyGradient(view_div, learning_rate, training_repeats)
    
# get the initial observations
observations, has_neighbor = aggr_env.get_observations()
# initialize variable for last statuses
observations_last = np.copy(observations)
has_neighbor_last = has_neighbor[:]

# the loop
sleep_time = 0.1
episode_threshold = 200  # threshold of number of samples to trigger a training
data_total = 0  # running total of training samples
while True:
    # decide actions base on observations
    actions = [0 for i in range(robot_quantity)]  # 0 for no turning as default
    for i in range(robot_quantity):
        if has_neighbor[i]:  # has neighbor, choose action based on nn output
            actions[i] = PG.choose_action(observations[i])
            # if has no neighbors, will use default 0 for no turning

    # update one step of actions in environment
    rewards = aggr_env.step_update_without_display(actions)
    print(rewards)
    # check tkinter window right before updating display
    if aggr_env.window_closed: break  # keep this like even if not updating display
    # will halt the program here if pause switch is on
    if need_pause:
        while aggr_env.pause_on:
            aggr_env.root.update()  # need this tk window update
    aggr_env.display_update()

    # get the observations after the actions have been taken
    observations, has_neighbor = aggr_env.get_observations()
    for i in range(robot_quantity):
        if has_neighbor_last[i] and rewards[i] != 0:  # avoid zero reward
            # store data for training as long as robot has neighbor before action
            PG.store_transition(observations_last[i], actions[i], rewards[i])
            data_total = data_total + 1
    # update last status of observations and has_neighbor
    observations_last = np.copy(observations)
    has_neighbor_last = has_neighbor[:]
    # the learning
    if data_total >= episode_threshold:
        data_total = 0  # reset running total of rewards
        PG.learn()

    # time.sleep(sleep_time)


