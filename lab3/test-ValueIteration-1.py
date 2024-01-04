import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self):
        terminal_view = '''
###################################################################################################################
#..................#..................#........................................................#..................#
#..................#..................#........................................................#..................#
#..................#..................#........................................................#..................#
#..................#..................#........................................................#..................#
#..................#..................#........................................................#..................#
#..................#..................#........................................................#..................#
#..................#..................#........................................................#..................#
#..................#..................#........................................................#..................#
#..................#..................#..................####################..................#..................#
#..................#.....................................#..................#..................#..................#
#..................#.....................................#..................#..................#..................#
#..................#.....................................#..................#..................#..................#
#..................#.....................................#..................#..................#..................#
#..................#.....................................#..................#..................#..................#
#..................#.....................................#..................#..................#..................#
#..................#.....................................#..................#..................#..................#
#..................#.....................................#..................#..................#..................#
#..................#.....................................#..................#..................#..................#
#..................####################..................#..................#..................#..................#
#.....................................#..................#........................................................#
#.....................................#..................#........................................................#
#.....................................#..................#........................................................#
#.....................................#..................#........................................................#
#.....................................#..................#........................................................#
#.....................................#..................#........................................................#
#.....................................#..................#........................................................#
#.....................................#..................#........................................................#
#.....................................#..................#........................................................#
#..................####################..................##########################################################
#........................................................#........................................................#
#........................................................#........................................................#
#........................................................#........................................................#
#........................................................#........................................................#
#........................................................#........................................................#
#........................................................#........................................................#
#........................................................#........................................................#
#........................................................#........................................................#
#........................................................#........................................................#
#..................#######################################..................####################..................#
#.....................................#.....................................#.....................................#
#.....................................#.....................................#.....................................#
#.....................................#.....................................#.....................................#
#.....................................#.....................................#.....................................#
#.....................................#.........B...........................#.....................................#
#.....................................#.....................................#.....................................#
#.....................................#.....................................#.....................................#
#.....................................#.....................................#.....................................#
#.....................................#.....................................#.....................................#
#..................#######################################..................####################..................#
#...........................................................................#.....................................#
#...........................................................................#.....................................#
#...........................................................................#.....................................#
#...........................................................................#.....................................#
#...........................................................................#..................G..................#
#...........................................................................#.....................................#
#...........................................................................#.....................................#
#...........................................................................#.....................................#
#...........................................................................#.....................................#
###################################################################################################################
'''

        grid_list = terminal_view.strip().split('\n')
        self.height = len(grid_list)
        self.width = len(grid_list[0])
        self.grid = np.zeros((self.height, self.width)) - 1

        for i, row in enumerate(grid_list):
            for j, cell in enumerate(row):
                if cell == '#':
                    self.grid[i, j] = -100
                elif cell == 'B':
                    self.bomb_location = (i, j)
                    self.grid[i, j] = -100
                elif cell == 'G':
                    self.gold_location = (i, j)
                    self.grid[i, j] = 250

                self.current_location = (np.random.randint(1, 59), np.random.randint(1, 9))
        self.terminal_states = [self.bomb_location, self.gold_location]

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

    def print_grid(self):
        grid_repr = ''
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.gold_location:
                    grid_repr += 'G'
                elif (i, j) == self.bomb_location:
                    grid_repr += 'B'
                elif self.grid[i, j] == -100:
                    grid_repr += '#'
                else:
                    grid_repr += '.'
            grid_repr += '\n'
        print(grid_repr)

    def get_available_actions(self):
        return self.actions

    def agent_on_map(self):
        grid = np.zeros(( self.height, self.width))
        grid[ self.current_location[0], self.current_location[1]] = 1
        return grid

    def get_reward(self, new_location):
        return self.grid[ new_location[0], new_location[1]]

    def make_step(self, action):
        last_location = self.current_location

        if action == 'UP':
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        elif action == 'DOWN':
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)

        elif action == 'LEFT':
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)

        elif action == 'RIGHT':
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)

        return reward
    
    def get_next_state_and_reward(self, state, action):
        if action == 'UP':
            next_state = (max(state[0] - 1, 0), state[1])
        elif action == 'DOWN':
            next_state = (min(state[0] + 1, self.height - 1), state[1])
        elif action == 'LEFT':
            next_state = (state[0], max(state[1] - 1, 0))
        elif action == 'RIGHT':
            next_state = (state[0], min(state[1] + 1, self.width - 1))
        reward = self.grid[next_state[0], next_state[1]]
        return next_state, reward

    def check_state(self):
        if self.current_location in self.terminal_states:
            return 'TERMINAL'

    def simulate_step(self, state, action):
        if action == 'UP':
            next_state = (max(state[0] - 1, 0), state[1])
        elif action == 'DOWN':
            next_state = (min(state[0] + 1, self.height - 1), state[1])
        elif action == 'LEFT':
            next_state = (state[0], max(state[1] - 1, 0))
        elif action == 'RIGHT':
            next_state = (state[0], min(state[1] + 1, self.width - 1))
        reward = self.get_reward(next_state)
        return next_state, reward

class ValueIterationAgent:
    def __init__(self, environment, discount=0.99, convergence_threshold=1e-6):
        self.environment = environment
        self.discount = discount
        self.convergence_threshold = convergence_threshold
        self.values = np.zeros((environment.height, environment.width))
        self.actions = environment.get_available_actions()

    def get_value(self, state, action):
        next_state, reward = self.environment.get_next_state_and_reward(state, action)
        return reward + self.discount * self.values[next_state]

    def run_value_iteration(self):
        while True:
            delta = 0
            for i in range(self.environment.height):
                for j in range(self.environment.width):
                    state = (i, j)
                    if state not in self.environment.terminal_states:
                        old_value = self.values[state]
                        new_value = max([self.get_value(state, action) for action in self.actions])
                        self.values[state] = new_value
                        delta = max(delta, abs(old_value - new_value))

            if delta < self.convergence_threshold:
                break

    def choose_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions)
        else:
            best_action = None
            best_value = float('-inf')
            for action in self.actions:
                action_value = self.get_value(state, action)
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            return best_action

def play(environment, agent, trials=2000, max_steps_per_episode=3000, learn=False, step_penalty=-0.1):
    reward_per_episode = []
    for trial in range(trials):
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and game_over != True:
            old_state = environment.current_location
            action = agent.choose_action(old_state)
            reward = environment.make_step(action)
            
            # Добавьте штраф за каждый шаг
            reward += step_penalty
            
            cumulative_reward += reward
            step += 1

            if environment.check_state() == 'TERMINAL':
                environment.__init__()
                game_over = True

        reward_per_episode.append(cumulative_reward)

    return reward_per_episode

env = GridWorld()
value_iteration_agent = ValueIterationAgent(env)
value_iteration_agent.run_value_iteration()

reward_per_episode = play(env, value_iteration_agent, trials=2000, max_steps_per_episode=3000, learn=False, step_penalty=-0.1)

plt.plot(reward_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

env = GridWorld()
env.print_grid()

