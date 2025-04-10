import matplotlib.pyplot as plt
import numpy as np
import random

class Env:
    def __init__(self, num_cars):
        self.fig = None  # Initialize as None
        self.ax = None
        self.weather = random.randint(0, 3)
        self.grids = self.adjust_env_for_weather(self.weather, (3, 3))
        self.num_cars = num_cars


    def adjust_env_for_weather(self, weather_severity, center):
        environment = np.ones((7, 7))
        adjustment_factors = {0: 0, 1: 1, 2: 2, 3: 3}
        max_distance = np.linalg.norm(np.array(center) - np.array([0, 0]))
        for i in range(7):
            for j in range(7):
                distance = np.linalg.norm(np.array([i, j]) - np.array(center))
                environment[i, j] += (1 - distance / max_distance) * adjustment_factors[weather_severity]
        return np.round(environment, 2)

    def adjust_congestion_around_car(self, car_position, congestion_map, adjustment=0.5):
        rows, cols = congestion_map.shape
        for i in range(max(0, car_position[0] - 1), min(rows, car_position[0] + 2)):
            for j in range(max(0, car_position[1] - 1), min(cols, car_position[1] + 2)):
                congestion_map[i, j] += adjustment
        return np.round(congestion_map, 2)

    # def step(self, act, car_id, cars, cars_position):
    #     state = cars[car_id].car_observation.copy()
    #     # Denormalize to get integer positions
    #     x, y = int(state[1] * 6), int(state[2] * 6)
    #     done = False
    #
    #     if cars[car_id].has_arrived:
    #         return state, 0, True
    #
    #     # Update position based on action
    #     if act == 0 and x > 0:  # Up
    #         x -= 1
    #     elif act == 1 and x < 6:  # Down
    #         x += 1
    #     elif act == 2 and y > 0:  # Left
    #         y -= 1
    #     elif act == 3 and y < 6:  # Right
    #         y += 1
    #
    #     # Denormalize destination
    #     dest_x, dest_y = int(state[3] * 6), int(state[4] * 6)
    #
    #     if x == dest_x and y == dest_y:
    #         reward = 10
    #         done = True
    #         cars[car_id].has_arrived = True
    #     elif (x, y) == (int(state[1] * 6), int(state[2] * 6)):
    #         reward = -0.5  # No movement
    #     else:
    #         reward = -self.grids[x, y] * 0.5
    #
    #     dist = np.linalg.norm(np.array([x, y]) - np.array([dest_x, dest_y]))
    #     reward += -dist * 0.1
    #
    #     # Update observation with new normalized position
    #     cars[car_id].car_observation[1] = x / 6.0
    #     cars[car_id].car_observation[2] = y / 6.0
    #     self.upgrade_env(cars, cars_position)
    #     return cars[car_id].car_observation, reward, done

    def step(self, act, car_id, cars, cars_position):
        state = cars[car_id].car_observation.copy()
        x, y = int(state[1] * 6), int(state[2] * 6)
        done = False

        if cars[car_id].has_arrived:
            return state, 0, True

        # Update position based on action
        if act == 0 and x > 0:
            x -= 1  # Up
        elif act == 1 and x < 6:
            x += 1  # Down
        elif act == 2 and y > 0:
            y -= 1  # Left
        elif act == 3 and y < 6:
            y += 1  # Right

        dest_x, dest_y = int(state[3] * 6), int(state[4] * 6)

        # Base reward
        if x == dest_x and y == dest_y:
            reward = 50
            done = True
            cars[car_id].has_arrived = True
        else:
            # Grid penalty
            reward = -self.grids[x, y] * 0.5
            # Shaping reward based on distance change
            prev_dist = np.linalg.norm(np.array([int(state[1] * 6), int(state[2] * 6)]) - np.array([dest_x, dest_y]))
            new_dist = np.linalg.norm(np.array([x, y]) - np.array([dest_x, dest_y]))
            shaping_reward = prev_dist - new_dist
            reward += shaping_reward
            # Congestion penalty
            congestion_penalty = 0
            for other_car_id in range(len(cars)):
                if other_car_id != car_id and not cars[other_car_id].has_arrived:
                    other_x, other_y = int(cars[other_car_id].car_observation[1] * 6), int(
                        cars[other_car_id].car_observation[2] * 6)
                    if np.linalg.norm(np.array([x, y]) - np.array([other_x, other_y])) < 2:
                        congestion_penalty -= 0.5
            reward += congestion_penalty

        # Update observation
        cars[car_id].car_observation[1] = x / 6.0
        cars[car_id].car_observation[2] = y / 6.0
        self.upgrade_env(cars, cars_position)
        return cars[car_id].car_observation, reward, done

    def upgrade_env(self, cars, cars_position):
        self.grids = self.adjust_env_for_weather(self.weather, (3, 3))
        for position in cars_position:
            self.grids = self.adjust_congestion_around_car(position, self.grids, adjustment=0.5)
        congestion_feature = self.grids.flatten().astype(np.float32)
        for car in cars:
            car.car_observation[5:] = congestion_feature

    def reset(self, cars, cars_position):
        cars_position.clear()
        self.weather = random.randint(0, 3)
        self.grids = self.adjust_env_for_weather(self.weather, (3, 3))

        for car_id in range(len(cars)):
            cars[car_id].has_arrived = False
            cars[car_id].init_position = [random.randint(start, end - 1) for start, end in
                                          [(0, 3), (0, 3), (5, 7), (5, 7)]]

            car_id_feature = np.array([car_id], dtype=np.float32)
            current_pos_feature = np.array(cars[car_id].init_position[:2], dtype=np.float32)
            destination_feature = np.array(cars[car_id].init_position[2:4], dtype=np.float32)
            congestion_feature = self.grids.flatten().astype(np.float32)

            cars[car_id].car_observation = np.concatenate([
                car_id_feature, current_pos_feature, destination_feature, congestion_feature
            ])
            cars[car_id].car_observation = self.normalize_state(cars[car_id].car_observation)
            cars_position.append(cars[car_id].init_position[:2])

        return cars, cars_position

    def render(self, cars):
        if self.fig is None:  # Create figure only on first render
            self.fig, self.ax = plt.subplots()
        self.ax.clear()  # Clear the axes to remove previous plots
        grid_display = (self.grids - self.grids.min()) / (self.grids.max() - self.grids.min() + 1e-10)
        self.ax.imshow(grid_display, cmap='hot_r', interpolation='nearest', vmin=0, vmax=0.8)
        for car in cars:
            car_x = round(car.car_observation[1] * 6)
            car_y = round(car.car_observation[2] * 6)
            self.ax.plot(car_y, car_x, 'go')
            dest_x = round(car.car_observation[3] * 6)
            dest_y = round(car.car_observation[4] * 6)
            self.ax.plot(dest_y, dest_x, 'ro')
        plt.draw()
        plt.pause(0.3)

    def normalize_state(self, state):
        state[1:5] = state[1:5] / 6.0  # 归一化坐标到 [0,1]
        grid_min = np.min(self.grids)
        grid_max = np.max(self.grids)
        state[5:] = (state[5:] - grid_min) / (grid_max - grid_min + 1e-10)
        return state