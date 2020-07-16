import tkinter as tk
import time
import threading
from math import *
import numpy as np

# Defined Constants
CART_WIDTH = 100
CART_HEIGHT = CART_WIDTH * 0.4
POLE_SIZE = 5
POLE_HEIGHT = CART_WIDTH
CANVAS_WIDTH = 8 * CART_WIDTH
CANVAS_HEIGHT = 8 * CART_HEIGHT
MAX_X = CANVAS_WIDTH - 3 * CART_WIDTH  # Two cart width from edge of screen. The 3 in MAX_X
MIN_X = 2 * CART_WIDTH  # is because cartX is on the left side of the cart
MAX_THETA = (15 / 360) * 2 * pi

# Physical Constants
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
# LENGTH = POLE_HEIGHT / 2.0  # Actually half the pole length
LENGTH = 50
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # Seconds between state updates
FOURTHIRDS = 1.333333333333

# max_theta_dot = 0.0
# max_x_dot = 0.0

runAnimation = False


class Cart:
    def __init__(self):
        self.cartX = 0.0
        self.cartY = CANVAS_HEIGHT - CART_HEIGHT
        self.cartX_dot = 0.0  # Cart's velocity
        self.theta = 0.0  # Angle of pole in relation to cart
        self.theta_dot = 0.0  # Angular velocity of the pole

        self.reset_state_vars()

    def reset_state_vars(self):
        self.cartX = (CANVAS_WIDTH / 2.0) - (CART_WIDTH / 2.0)
        self.cartX_dot = 0.0  # Cart's velocity
        self.theta = 0.0  # Angle of pole in relation to cart
        self.theta_dot = 0.0  # Angular velocity of the pole

    def pole_coords(self):
        x1 = self.cartX + (CART_WIDTH / 2)
        y1 = self.cartY
        x2 = x1 + POLE_HEIGHT * sin(self.theta)
        y2 = y1 - POLE_HEIGHT * cos(self.theta)

        return x1, y1, x2, y2

    def move_cart(self, direction):
        """
        Updates states variables based on cart movement
        :param direction: -1 is backwards, 0 is stationary, 1 is forward
        :return:
        """

        # Set force based on direction
        if direction == -1:
            force = -FORCE_MAG
        elif direction == 1:
            force = FORCE_MAG
        else:
            force = 0

        # ###  Calculate physical variables needed for update   ### #
        costheta = cos(self.theta)
        sintheta = sin(self.theta)

        temp = (force + POLEMASS_LENGTH * self.theta_dot * self.theta_dot * sintheta) / TOTAL_MASS

        theta_acc = (GRAVITY * sintheta - costheta * temp) / \
                    (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS))

        x_acc = temp - POLEMASS_LENGTH * theta_acc * costheta / TOTAL_MASS

        # ###  Update state variables  ### #
        self.cartX += (TAU * self.cartX_dot)
        self.cartX_dot += (TAU * x_acc)
        self.theta += (TAU * self.theta_dot)
        self.theta_dot += (TAU * theta_acc)

        # ###  Check Fail Conditions and Set Reward### #
        done = self.cartX < MIN_X \
               or self.cartX > MAX_X \
               or self.theta < -MAX_THETA \
               or self.theta > MAX_THETA

        if done:
            reward = -5
        else:
            reward = 0
        return reward, done


class InitialInput:
    def __init__(self):
        self.root = tk.Tk()

        # Create Entry Widgets
        self.episodes = tk.Entry(self.root)
        self.max_i = tk.Entry(self.root)
        self.discount = tk.Entry(self.root)
        self.epsilon = tk.Entry(self.root)
        self.eps_min = tk.Entry(self.root)
        self.eps_decay = tk.Entry(self.root)
        self.anim_frames = tk.Entry(self.root)

        # Set initial value of Entry Widgets
        self.episodes.insert(10, "100")
        self.max_i.insert(10, "1000")
        self.discount.insert(10, "0.9")
        self.epsilon.insert(10, "1.0")
        self.eps_min.insert(10, "0.1")
        self.eps_decay.insert(10, "0.9")
        self.anim_frames.insert(10, "10")

        self.make_form()
        self.root.mainloop()

    def make_form(self):
        tk.Label(self.root, text="Episodes: ").grid(row=0)
        tk.Label(self.root, text="Max Iterations: ").grid(row=1)
        tk.Label(self.root, text="Discount Factor: ").grid(row=2)
        tk.Label(self.root, text="Initial Epsilon: ").grid(row=3)
        tk.Label(self.root, text="Minimum Epsilon: ").grid(row=4)
        tk.Label(self.root, text="Epsilon Decay: ").grid(row=5)
        tk.Label(self.root, text="Show animation every x episodes").grid(row=6)

        self.episodes.grid(row=0, column=1)
        self.max_i.grid(row=1, column=1)
        self.discount.grid(row=2, column=1)
        self.epsilon.grid(row=3, column=1)
        self.eps_min.grid(row=4, column=1)
        self.eps_decay.grid(row=5, column=1)
        self.anim_frames.grid(row=6, column=1)

        tk.Button(self.root, text="Start", command=self.start).grid(row=7, column=0)

    def start(self):
        episodes = int(self.episodes.get())
        max_i = self.max_i.get()
        max_i = int(max_i)
        discount = float(self.discount.get())
        epsilon = float(self.epsilon.get())
        eps_min = float(self.eps_min.get())
        eps_decay = float(self.eps_decay.get())
        anim_frames = self.anim_frames.get()
        anim_frames = int(anim_frames)

        self.root.destroy()

        x = threading.Thread(target=thread_func)
        x.start()

        q = QLearn(episodes=episodes, max_iterations=max_i, state_limits=(1, 1, 7, 3,), discount=discount,
                   epsilon_min=eps_min, epsilon=epsilon, epsilon_decay=eps_decay, anim_frames=anim_frames)
        q.run()


class Animation:
    def __init__(self, master):
        self.root = master
        self.canvas = tk.Canvas(self.root,
                                width=CANVAS_WIDTH,
                                height=CANVAS_HEIGHT)
        self.draw_grid()
        self.cart = self.canvas.create_rectangle(cart.cartX,
                                                 cart.cartY,
                                                 cart.cartX + CART_WIDTH,
                                                 CANVAS_HEIGHT,
                                                 fill="red")

        self.pole = self.canvas.create_line(cart.pole_coords(),
                                            fill="black",
                                            width=POLE_SIZE)
        self.canvas.pack()

    def horizontal_line(self, y):
        self.canvas.create_line(0, y, CANVAS_WIDTH, y, fill="#aaaaaa", width=1)

    def vertical_line(self, x):
        self.canvas.create_line(x, 0, x, CANVAS_HEIGHT, fill="#aaaaaa", width=1)

    def draw_grid(self):
        ymid = CANVAS_HEIGHT / 2
        xmid = CANVAS_WIDTH / 2

        self.vertical_line(xmid)
        self.horizontal_line(ymid)
        line_distance = CART_WIDTH / 2
        count = 1
        while (xmid - count * line_distance) > 0:
            self.vertical_line(int(xmid - count * line_distance))
            self.vertical_line(int(xmid + count * line_distance))
            count += 1

        count = 1
        while (ymid - count * line_distance) > 0:
            self.horizontal_line((ymid - count * line_distance))
            self.horizontal_line((ymid + count * line_distance))
            count += 1

    def animate(self):
        if runAnimation:
            self.canvas.coords(self.cart,
                               (cart.cartX,
                                cart.cartY,
                                (cart.cartX + CART_WIDTH),
                                CANVAS_HEIGHT))
            self.canvas.coords(self.pole, (cart.pole_coords()))
        self.root.after(15, self.animate)


class QLearn:
    def __init__(self, episodes=100, max_iterations=1500, state_limits=(1, 1, 7, 3,), discount=0.9, epsilon_min=0.25,
                 epsilon=1.0, epsilon_decay=0.995, anim_frames=10):
        self.episodes = episodes
        self.max_iterations = max_iterations
        self.state_limits = state_limits
        self.gamma = discount
        self.actions = [-1, 1]
        self.num_actions = len(self.actions)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_x_dot = 0.0
        self.max_theta_dot = 0.0
        self.anim_frames = anim_frames

        # Multideminsional array for each of the 4 state values, the number of
        # actions and times the state has been visited.
        self.Q = np.zeros(self.state_limits + (self.num_actions, ))
        self.visits = np.zeros(self.state_limits + (self.num_actions, ))
        self.set_max_velocity()

    def set_max_velocity(self):
        while cart.cartX > MIN_X:
            cart.move_cart(-1)
            if abs(cart.cartX_dot) > self.max_x_dot:
                self.max_x_dot = abs(cart.cartX_dot)
            if abs(cart.theta_dot) > self.max_theta_dot:
                self.max_theta_dot = abs(cart.theta_dot)

        cart.reset_state_vars()

    def choose_action_index(self, state, ep):
        epsilon = max(self.epsilon_min, min(self.epsilon, 1.0 - log10((ep + 1)*self.epsilon_decay)))

        if np.random.random() <= epsilon:  # ep + 1 since ep starts at 0
            # Generate random integer [0,num_states)
            i = int(np.random.random() * self.num_actions)
            # print(i)
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

    def get_state(self):
        # Bounds = [ x, x_dot, theta, theta_dot ]
        upper_bound = [MAX_X, self.max_x_dot, MAX_THETA, self.max_theta_dot]
        lower_bound = [MIN_X, -self.max_x_dot, -MAX_THETA, -self.max_theta_dot]
        state_vars = (cart.cartX, cart.cartX_dot, cart.theta, cart.theta_dot, )
        # Find ration of current state vars in relation to to their range
        ratios = []
        state = []
        for i in range(len(state_vars)):
            ratios.append((state_vars[i] + abs(lower_bound[i])) / (upper_bound[i] - lower_bound[i]))
            # Find state index by multiplying state ratios by state limits and rounding
            state.append(round((self.state_limits[i] - 1) * ratios[i]))
            # If values somehow get out of the bounds, set state index to 0 or max index
            state[i] = min(self.state_limits[i] - 1, max(0, state[i]))
        return tuple(state)

    def update_q(self, old_state, action, reward, new_state):
        alpha = 1 / (1 + self.visits[old_state][action])
        self.visits[old_state][action] += 1
        self.Q[old_state][action] = (1 - alpha) * self.Q[old_state][action] + \
            alpha * (reward + self.gamma * np.max(self.Q[new_state]))

        # print(self.Q[old_state][action])

    def run(self):
        global runAnimation

        episode_complete = 0
        balanced_count = 0
        most_iterations = 0
        for ep in range(self.episodes):
            cart.reset_state_vars()
            current_state = self.get_state()
            done = False
            if self.anim_frames == 0:
                runAnimation = False
            elif ep % self.anim_frames == 0:
                runAnimation = True
            else:
                runAnimation = False
            i = 0
            while not done:
                i += 1

                act_i = self.choose_action_index(current_state, ep)
                # print(act)
                if act_i not in (0, 1, 2):
                    exit()
                # print(self.actions[act_i])
                reward, done = cart.move_cart(self.actions[act_i])
                new_state = self.get_state()
                self.update_q(current_state, act_i, reward, new_state)
                current_state = new_state
                if runAnimation:
                    time.sleep(TAU)
                    if i % 100 == 0:
                        print("Executing iteration " + str(i) + " of episode " + str(ep))

                if done:
                    if i > most_iterations:
                        most_iterations = i

                if i > self.max_iterations:
                    balanced_count += 1
                    print("Max iterations reached at episode " + str(ep) + "! Moving on!")
                    if episode_complete == 0:
                        episode_complete = ep
                    break

        if episode_complete == 0:
            print("Never balanced")
            print("Most iterations: " + str(most_iterations))
        else:
            print("First episode balanced: " + str(episode_complete))
        # print(self.Q)


def thread_func():
    master = tk.Tk()
    animation = Animation(master)
    animation.animate()
    master.mainloop()


cart = Cart()
start = InitialInput()


#
# q = QLearn()
# q.run()

# runAnimation = True
#
# for i in range(75):
#     move_cart(1)
#     time.sleep(TAU)
#
# for i in range(300):
#     move_cart(-1)
#     time.sleep(TAU)


#
# runAnimation = False
