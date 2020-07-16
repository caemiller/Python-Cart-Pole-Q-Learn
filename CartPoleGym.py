import numpy as np
import gym
from math import *
from collections import deque


class CartPoleGym:
    def __init__(self, episodes=1000, win_iterations=195, state_limits=(1, 1, 7, 3,), discount=0.9, epsilon_min=0.1,
                 epsilon=1.0, epsilon_decay=0.9):
        self.env = gym.make('CartPole-v0')
        self.episodes = episodes
        self.win_iterations = win_iterations
        self.state_limits = state_limits
        self.gamma = discount
        self.num_actions = self.env.action_space.n
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_x_dot = 50.0
        self.max_theta_dot = 1.0

        # Multideminsional array for each of the 4 state values, the number of
        # actions and times the state has been visited.
        self.Q = np.zeros(self.state_limits + (self.num_actions, ))
        self.visits = np.zeros(self.state_limits + (self.num_actions, ))

    def choose_action(self, state, ep):
        epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - log10((ep + 1)*self.epsilon_decay)))

        if np.random.random() <= epsilon:  # ep + 1 since ep starts at 0
            # Generate random integer [0,num_states)
            i = self.env.action_space.sample()
            return i
        else:
            max_i = np.argmax(self.Q[state])
            temp = np.where(self.Q[state] == self.Q[state][max_i])[0]
            if len(temp) == 1:
                return max_i
            else:
                i = int(np.random.random() * len(temp))
                act = temp[i]
                return act

    def get_state(self, state_vars):
        obs_space = self.env.observation_space
        # Bounds = [ x, x_dot, theta, theta_dot ]
        upper_bound = [obs_space.high[0], self.max_x_dot, obs_space.high[2], self.max_theta_dot]
        lower_bound = [obs_space.low[0], -self.max_x_dot, obs_space.low[2], -self.max_theta_dot]

        # Find ratio of current state vars in relation to to their range
        ratios = []
        state = []
        for i in range(len(state_vars)):
            ratios.append((state_vars[i] + abs(lower_bound[i])) / (upper_bound[i] - lower_bound[i]))
            # Find state index by multiplying state ratios by state limits and rounding
            state.append(round((self.state_limits[i] - 1) * ratios[i]))
            # If values somehow get out of the bounds, set state index to 0 or max index
            state[i] = int(min(self.state_limits[i] - 1, max(0, state[i])))
        return tuple(state)

    def update_q(self, old_state, action, reward, new_state):
        alpha = 1 / (1 + self.visits[old_state][action])
        self.visits[old_state][action] += 1
        self.Q[old_state][action] = (1 - alpha) * self.Q[old_state][action] + \
            alpha * (reward + self.gamma * np.max(self.Q[new_state]))

        # print(self.Q[old_state][action])

    def run(self):
        scores = deque(maxlen=100)
        # episode_complete = 0
        # balanced_count = 0
        # most_iterations = 0
        for ep in range(self.episodes):

            current_state = self.get_state(self.env.reset())
            done = False
            i = 0
            while not done:
                i += 1

                act = self.choose_action(current_state, ep)

                state_vars, reward, done, _ = self.env.step(act)
                new_state = self.get_state(state_vars)
                self.update_q(current_state, act, reward, new_state)
                current_state = new_state

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.win_iterations and ep >= 100:
                print('Solver after ' + str(ep-100) + " trials")
            if ep % 100 == 0:
                print("Episode: " + str(ep) + " - Average score: " + str(mean_score))
        print('Did not solve after {} episodes 😞'.format(ep))


solver = CartPoleGym()
solver.run()
