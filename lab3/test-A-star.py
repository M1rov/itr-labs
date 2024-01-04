import numpy as np
import matplotlib.pyplot as plt
import heapq

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

                self.current_location = (np.random.randint(51, 59), np.random.randint(1, 9))
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

def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def a_star_search(gridworld, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not len(frontier) == 0:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for action in gridworld.actions:
            next_state, _ = gridworld.simulate_step(current, action)
            new_cost = cost_so_far[current] + 1

            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + manhattan_distance(next_state, goal)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current

    return reconstruct_path(came_from, start, goal)

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def play_with_a_star(env, trials=2000, max_steps_per_episode=3000, step_penalty=-0.1):
    rewards_per_episode = []

    for _ in range(trials):
        env.__init__()  # Reinitialize the environment for each trial
        path = a_star_search(env, env.current_location, env.gold_location)
        total_reward = 0

        for step in range(min(len(path), max_steps_per_episode)):
            if step < len(path) - 1:
                current = path[step]
                next_step = path[step + 1]
                action = None
                if next_step[0] < current[0]: action = 'UP'
                elif next_step[0] > current[0]: action = 'DOWN'
                elif next_step[1] < current[1]: action = 'LEFT'
                elif next_step[1] > current[1]: action = 'RIGHT'
                reward = env.make_step(action)
                total_reward += reward + step_penalty
                if env.check_state() == 'TERMINAL':
                    break

        rewards_per_episode.append(total_reward)

    return rewards_per_episode

# Створення середовища та виконання гри з A*
env = GridWorld()
rewards_per_episode = play_with_a_star(env, trials=2000, max_steps_per_episode=3000)

# Візуалізація результатів
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

env = GridWorld()
env.print_grid()

