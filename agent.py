import random
from model import PPO  # 导入 PPO 类

class Car:
    def __init__(self, car_id):
        self.car_id = car_id
        self.action_frequencies = {0: 0, 1: 0, 2: 0, 3: 0}
        self.policy = PPO()
        self.init_position = [random.randint(start, end - 1) for start, end in [(0, 3), (0, 3), (5, 7), (5, 7)]]
        self.car_observation = []
        self.has_arrived = False

    def choose_action(self):
        if len(self.car_observation) != 54:
            print(f"Warning: state dimension is {len(self.car_observation)}, expected 54")
        action, _, _ = self.policy.choose_action(self.car_observation)
        self.action_frequencies[action] += 1
        # print(f"Car {self.car_id} chose action {action}, frequencies: {self.action_frequencies}")
        return action