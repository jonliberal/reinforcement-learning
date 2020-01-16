import random
import gym
import numpy as np

import time


# create cart pole environment
ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME)
env._max_episode_steps = 20000
# get main parameters from the agent-environment interaction

number_of_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

# Model parameters
H = 8
A = number_of_actions
S = state_dim

model = {}
model['W1'] = [[-0.29000811, -2.35978436, -2.2353414,  -1.2999726 ],
 [-0.16721503 , 2.80432466 , 3.20868981 , 1.64545463],
 [-0.4474617 , -2.00660431 ,-1.70652474 ,-1.42049036],
 [-0.92481043 ,-0.96298212 ,-1.20088074 ,-1.16880308],
 [-0.76706268 ,-0.65544091, -0.44535123 ,-1.08063228],
 [-0.42027842 ,-0.87492293,  0.64088356 , 1.48973643],
 [-0.90846624 ,-0.66730323, -0.6571431  ,-1.10098213],
 [-0.90904136 ,-1.16572142, -1.26671721 ,-1.45497671]]

model['W2'] = [[ 3.1060679 , -3.43533481 , 2.66553673 , 1.98811831 , 1.05842792 ,-1.65759528
  , 1.6790153 ,  2.84166548],
 [-3.22629681 , 2.65039459 ,-2.98427899 ,-1.93152952 ,-1.42733581 , 1.15693004
,  -1.30790112, -2.04708596]]




def softmax(y):
    if max(y) > 10:
        print("WARNING: NUMEROS GORDACOS")
        print(y)
    return np.exp(y)/np.sum(np.exp(y))

def propagate_forward(x):
    global model
    h = np.dot(model['W1'], x)
    h[h<0] = 0
    y = np.dot(model['W2'], h)
    p = softmax(y)

    return h,y,p



for e in range(1000):
    terminal = False
    state = env.reset()
    steps = 0
    while not terminal:
        steps+= 1
        env.render()


        h,y,p = propagate_forward(state)




        # select action for current state
        action = np.random.choice(2, p=p)

        state, reward, terminal, info = env.step(action)
    print(steps)
    time.sleep(3)
