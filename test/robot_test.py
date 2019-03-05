import sys
import os
import time

# Hacky but tacky
sys.path.append(os.getcwd())

import robotic_warehouse.robotic_warehouse as rw
import kex_robot.robot as KR

capacity = 20
timestamp = time.time()
gym = rw.RoboticWarehouse(
    robots=40,
    capacity=capacity,
    spawn=1000,
    shelve_length=10,
    shelve_height=10,
    shelve_width=30,
    shelve_throughput=1,
    cross_throughput=2,
    periodicity_lower=500,
    periodicity_upper=1000,
    seed=105)
print("Setup Time: {}".format(time.time() - timestamp))

steps = 0
timestamp = time.time()

robots, packages = gym.reset()

R = []
reservations = set()

for r in robots:
    R.append(KR.Robot(r, gym, capacity, reservations, R))

try:
    while True:
        gym.render()
        actions = [r(packages) for r in R]
        (_, packages), _, _, _ = gym.step(actions)

        steps += 1
        # time.sleep(1.0)
except KeyboardInterrupt:
    print("Number of steps: {}, average step per second: {}".format(
        steps, steps / (time.time() - timestamp)))
