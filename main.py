from env import Env
from agent import Car
from train import train
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    num_cars = 5
    env = Env(num_cars)
    cars = [Car(car_id) for car_id in range(num_cars)]

    # 训练
    train(cars, env, episode=500, num_cars=num_cars)

    # 评估（可选）
    # from eval import eval
    # eval(cars, env)