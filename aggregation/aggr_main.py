# main program for the aggregation simulation with policy gradient learning

# interact with the environment, and brain of reinforcement learning


from aggr_env import AggrEnv
import time

sim_env = AggrEnv(30)

sleep_time = 0.1

while True:
    for _ in range(20):
        sim_env.frame_update(5,0)
        time.sleep(sleep_time)
    for _ in range(20):
        sim_env.frame_update(0,5)
        time.sleep(sleep_time)
    for _ in range(20):
        sim_env.frame_update(-5,0)
        time.sleep(sleep_time)
    for _ in range(20):
        sim_env.frame_update(0,-5)
        time.sleep(sleep_time)

