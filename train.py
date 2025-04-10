import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def train(cars, env, episode=500, num_cars=2):
    episode_rewards = [[] for _ in range(num_cars)]

    for i in tqdm(range(episode)):
        cars, cars_position = env.reset(cars, [])
        individual_episode_rewards = [0] * num_cars
        done = False
        step_num = 0
        while not done:
            all_done = True
            for car_id in range(num_cars):
                if not cars[car_id].has_arrived:
                    s = cars[car_id].car_observation
                    a, v, action_prob = cars[car_id].policy.choose_action(s)
                    s_, r, car_done = env.step(a, car_id, cars, cars_position)
                    cars[car_id].policy.store_transition(s, a, r, s_, car_done, v, action_prob)
                    individual_episode_rewards[car_id] += r
                    if car_done:
                        cars[car_id].has_arrived = True
                    else:
                        all_done = False
            if all_done:
                done = True
            step_num += 1
            if step_num >= 2000:
                break

        for car_id in range(num_cars):
            cars[car_id].policy.learn()

        for idx, reward in enumerate(individual_episode_rewards):
            episode_rewards[idx].append(reward)

    # 保存模型
    for car_id in range(num_cars):
        torch.save(cars[car_id].policy.policy_net.state_dict(), f'weight/car_{car_id}_policy.pth')
        print(f"Saved model for car {car_id}")

    # 绘制奖励曲线
    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    window_size = 10
    plt.figure(3)
    for idx, rewards in enumerate(episode_rewards):
        if len(rewards) >= window_size:
            smoothed_rewards = moving_average(rewards, window_size)
            episodes = np.arange(window_size - 1, len(rewards))
            plt.plot(episodes, smoothed_rewards, label=f'Agent {idx + 1} Smoothed Rewards')
        else:
            plt.plot(range(len(rewards)), rewards, label=f'Agent {idx + 1} Rewards')
    plt.title('Smoothed Total Rewards per Agent per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    plt.show()