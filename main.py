# Script for deciding what proportion of salmons
# to catch in a year in a specific area
# and maximizing the longer term return.

import mdptoolbox.example as mdpExample
import mdptoolbox
import numpy as np

# The number of states
STATES = 4

# The number of actions
ACTIONS = 2
ACTION_FISH = 0
ACTION_DO_NOT_FISH = 1

# P[0] = fish
# Probability array
P = np.array([
    # P[0] = fish
    [[0.75, 0.25, 0, 0],  # 0 = empty/
     [0, 0.75, 0.25, 0],  # 1 = low/
     [0, 0, 0.6, 0.4],  # 2 = medium
     [0, 0, 0, 1]],  # 3 high

    # P[1] = not to fish
    [[0, 1, 0, 0],  # 0 = empty
     [0, 0.3, 0.7, 0],  # 1 = low
     [0, 0, 0.25, 0.75],  # 2 = medium
     [0, 0, 0.05, 0.95]]  # 3 high
])
# actions array
# [0] = not to fish
# [1] = fish
R = np.array([[-200, 10],  # empty
              [0, 5],  # low
              [0, 10],  # medium
              [0, 50]])  # high

print("P=", P)
print("R=", R)

# rewards?
Discount = 0.9
NumPeriods = 10

##########################
print("Value Iteration")
vi = mdptoolbox.mdp.ValueIteration(P, R, Discount, NumPeriods)
vi.setVerbose()
vi.run()
print("optimal value function=", vi.V)
print("optimal policy=", vi.policy)
##########################

##########################
print("Policy Iteration")
pi = mdptoolbox.mdp.PolicyIteration(P, R, Discount)
pi.setVerbose()
pi.run()
print("optimal value function=", pi.V)
print("optimal policy=", pi.policy)
##########################
