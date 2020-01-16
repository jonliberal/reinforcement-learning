import random
import gym
import numpy as np
from collections import deque

# create cart pole environment
ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME)

# get main parameters from the agent-environment interaction

number_of_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

# Model parameters
H = 8
A = number_of_actions
S = state_dim

# Model weights
model = {}
model['W1'] = np.random.randn(H,S) / np.sqrt(S) # "Xavier" initialization
model['W2'] = np.random.randn(A,H) / np.sqrt(H)

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

def policy_gradient(action, reward, y, p):
    grad = -p
    grad[action] += 1.
    return grad*reward

def backprop(dy, h, x):
    global model
    # we are lucky that as we have dense layers, the gradient is linear

    # W2 update
    dw2 = np.outer(dy, h)
    ## that means that if we ONLY update W2 += dw2, now forward propagating
    ## the same input x will yield the 'perfect' policy value for this example (proof below)
    ## W2 ready :)

    # W1 update
    dh = np.dot(dy, model['W2'])
    dh[h<=0] = 0
    ## now we have how much we need to increase the hidden state: h = h+lr*dh
    dw1 = np.outer(dh, x)
    ## W1 ready :)
    return dw1, dw2


gamma = 0.97
gamma_series = [1]
for e in range(1000):
    gamma_series.append(gamma_series[-1]*gamma + 1)


model = {}
model['W1'] = np.random.randn(H,S) / np.sqrt(S*H) # "Xavier" initialization
model['W2'] = np.random.randn(A,H) / np.sqrt(H*A)

learning_rate = 0.0001
count_episodes = 0

gradient_buffer = []







for i in range(1000):
# Play a bunch of episodes
    rewards = []
    states = []
    actions = []
    position_rewards= []


    for e in range(100):
        terminal = False
        state = env.reset()
        r = 0

        while not terminal:
            if  count_episodes %300==0:
                env.render()


            h,y,p = propagate_forward(state)

            states.append(state)



            # select action for current state
            action = np.random.choice(2, p=p)
            actions.append(action)

            state, reward, terminal, info = env.step(action)
            # we want to keep the pole close to the center of the screen
            position_rewards.append(-np.abs(state[0])*2)

            r+= 1


        count_episodes+=1

        if  count_episodes %10000==0:
            print(model['W1'])
            print(model['W2'])

        rewards.append(r)

    print('Average episode lasted: ',np.mean(rewards))
    print('Number of episodes played: ',count_episodes)
    rs = [list(reversed(gamma_series[:r])) for r in rewards]
    mean_r = np.mean([np.mean(r) for r in rs])
    rs = np.concatenate(rs) + np.array(position_rewards)
    normalized_rewards = (rs - np.mean(rs))/np.std(rs)


    # train

    ind = np.arange(len(rs))
    np.random.shuffle(ind)
    for t in ind:
        reward = normalized_rewards[t]
        state = states[t]
        action = actions[t]

        h,y,p = propagate_forward(state)
        dy = policy_gradient(action,reward, y, p)
        dw1, dw2 = backprop(dy, h, state)

        #update weights (train)
        w1_update = np.clip(learning_rate * dw1, -0.0005, +0.0005)
        w2_update = np.clip(learning_rate * dw2, -0.0005, +0.0005)

        model['W1'] += w1_update
        model['W2'] += w2_update

    #gradient_buffer.append([np.linalg.norm(learning_rate * dw1), np.linalg.norm(learning_rate * dw2)])
    #gradient_buffer.append([np.max(np.abs(learning_rate * dw1)), np.max(np.abs(learning_rate * dw2))])
    #print(np.linalg.norm(dw1))
    #print(np.linalg.norm(dw2))
