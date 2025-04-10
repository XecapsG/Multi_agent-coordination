import torch
import numpy as np
from agent import Car
from env import Env
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation  # Import the animation module

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
num_cars = 5
cars_position = []
cars = []
env = Env(num_cars)

# Create cars and initialize positions
for car_id in range(num_cars):
    car = Car(car_id)
    car_id_feature = np.array([car_id], dtype=np.float32)
    car_position_feature = np.array(car.init_position[:2], dtype=np.float32)
    car.car_observation = np.concatenate([car_id_feature, car_position_feature])
    cars_position.append(car.init_position[:2])
    cars.append(car)

# Load model weights
for i in range(num_cars):
    filename = f'weight/car_{i}_policy.pth'
    cars[i].policy.policy_net.load_state_dict(torch.load(filename))
    cars[i].policy.policy_net.eval()

cars, cars_position = env.reset(cars, cars_position)

done = False
step_count = 0
max_steps = 50

# Set up the figure and writer
fig, ax = plt.subplots()
env.fig = fig  # ensure your env.render uses this figure
env.ax = ax
writer = animation.FFMpegWriter(fps=2)  # adjust fps as needed

# Initialize the rendering
env.render(cars)
plt.draw()

# Save the animation to a video file
with writer.saving(fig, "simulation.mp4", dpi=100):
    while not done and step_count < max_steps:
        all_done = True
        for car_id in range(num_cars):
            if not cars[car_id].has_arrived:
                state_before = cars[car_id].car_observation.copy()
                action = cars[car_id].choose_action()
                state_after, reward, car_done = env.step(action, car_id, cars, cars_position)
                cars_position[car_id] = [int(state_after[1] * 6), int(state_after[2] * 6)]
                env.render(cars)         # re-render the updated state
                plt.gcf().canvas.draw()   # update the canvas
                writer.grab_frame()       # capture this frame for the video
                print(f"Car {car_id} moved to: ({cars[car_id].car_observation[1]*6:.2f}, {cars[car_id].car_observation[2]*6:.2f})")
                if not car_done:
                    all_done = False
        step_count += 1
        if all_done:
            done = True

plt.close(fig)
