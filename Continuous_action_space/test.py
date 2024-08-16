#THs code tests the trained model saved from training.
# THe code tests for 5000 episodes and displays the graphs.

import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from highway_env.vehicle.controller import MDPVehicle

# Lists to store test results
test_rewards = []
test_merges = []
episodes_reward = []
class CustomMergeEnv(gym.Env):
    def __init__(self, render_mode='human'):
        super().__init__()
        self.env = gym.make("merge-v0", render_mode=render_mode)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = self.env.observation_space
        self.simulation_time = 0
        self.initial_speed = 10
        self.target_speed = 30
        self.merging_distance = 280
        self.merging_time = 0.75
        self.speed_limit = 40
        self.acceleration_range = (0, 6)
        self.deceleration_range = (-3, 0)
        self.goal_position = np.array([280, 0])
        self.viewer = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.unwrapped.np_random, _ = gym.utils.seeding.np_random(seed)
        obs, info = self.env.reset()
        self.env.unwrapped.road.vehicles = []
        road = self.env.unwrapped.road
        lane_index = ("j", "k", 0)
        ego_vehicle = MDPVehicle(
            road,
            road.network.get_lane(lane_index).position(0, 0),
            speed=self.initial_speed,
            target_speed=self.target_speed,
            route=[("j", "k", 0), ("k", "b", 0)]
        )
        road.vehicles.append(ego_vehicle)
        self.env.unwrapped.vehicle = ego_vehicle
        self.simulation_time = 0
        self.episode_reward = 0
        return self.env.unwrapped.observation_type.observe(), {}

    def step(self, action):
        ego_vehicle = self.env.unwrapped.vehicle
        previous_speed = ego_vehicle.speed

        if ego_vehicle.speed > self.speed_limit:
            ego_vehicle.speed = self.speed_limit

        acceleration = -3 + 4.5 * (action[0] + 1)

        if acceleration > 0:
            ego_vehicle.target_speed = min(self.speed_limit, ego_vehicle.speed + acceleration)
            action_type = "Acceleration"
        elif acceleration < 0:
            ego_vehicle.target_speed = max(0.1, ego_vehicle.speed + acceleration)
            action_type = "Deceleration"
        else:
            ego_vehicle.target_speed = ego_vehicle.speed
            action_type = "Maintain"

        ego_vehicle.step(1 / self.env.unwrapped.config["simulation_frequency"])

        obs, reward, terminated, truncated, info = self.env.step(0)

        self.simulation_time += 1 / self.env.unwrapped.config["simulation_frequency"]

        info['position'] = 'motorway' if ego_vehicle.position[0] >= self.merging_distance else 'ramp'
        info['previous_speed'] = previous_speed
        info['time_elapsed'] = self.simulation_time
        info['current_speed'] = ego_vehicle.speed
        info['successful_merge'] = False

        custom_reward = self.calculate_reward(action, info, ego_vehicle)
        self.episode_reward += custom_reward

        if ego_vehicle.position[0] >= self.goal_position[0]:
            terminated = True
            if self.target_speed - 0.5 <= ego_vehicle.speed <= self.target_speed + 0.5 and self.merging_time - 0.05 <= info['time_elapsed'] <= self.merging_time + 0.05:
                info['successful_merge'] = True

        if terminated or truncated:
            print(f"Episode ended. Merging speed: {ego_vehicle.speed:.2f} m/s, Time taken: {info['time_elapsed']:.2f} seconds, Reward: {self.episode_reward:.2f}")
            test_merges.append(1 if info['successful_merge'] else 0)
            test_rewards.append(self.episode_reward)
            print()
        return obs, self.episode_reward, terminated, truncated, info

    def calculate_reward(self, action, info, ego_vehicle):
        reward = 0

        if ego_vehicle.crashed:
            reward -= 10

        time_penalty = ((self.merging_time - info['time_elapsed'])/self.merging_time)**2
        speed_penalty = ((self.target_speed - ego_vehicle.speed)/self.target_speed)**2

        if ego_vehicle.position[0] >= self.goal_position[0]:
            if self.target_speed - 0.5 <= ego_vehicle.speed <= self.target_speed + 0.5 and self.merging_time - 0.05 <= info['time_elapsed'] <= self.merging_time + 0.05:
                print("Perfect Merging")
                reward += 10
                info['successful_merge'] = True  
            else:
                reward -= 2
                print("Imperfect Merging")
                if ego_vehicle.speed < self.target_speed - 0.5 or ego_vehicle.speed > self.target_speed + 0.5:
                    reward -= speed_penalty
                if info['time_elapsed'] > self.merging_time + 0.05 or info['time_elapsed'] < self.merging_time - 0.05:
                    reward -= time_penalty
                info['successful_merge'] = False
        else:
            reward -= 0.0001

        if ego_vehicle.speed > self.speed_limit or ego_vehicle.speed <= 10:
            reward -= 1

        if action[0] > 0 and ego_vehicle.speed - info['previous_speed'] > self.acceleration_range[1]:
            reward -= 1
        if action[0] < 0 and ego_vehicle.speed - info['previous_speed'] < self.deceleration_range[0]:
            reward -= 1

        return reward

    def render(self):
        frame = self.env.render()
        if frame is None:
            print("Warning: Rendered frame is None")
        return frame

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.env.close()

# Load the trained model
model = PPO.load("ramp_merging_model")

test_vec_env = make_vec_env(lambda: Monitor(CustomMergeEnv(render_mode='rgb_array'), info_keywords=('position', 'time_elapsed', 'current_speed')), n_envs=1)
test_episodes = 5000

# Run test episodes
for episode in range(test_episodes):
    obs = test_vec_env.reset()
    episode_reward = 0
    terminated = False
    while not terminated:
        action, _states = model.predict(obs)
        obs, rewards, dones, infos = test_vec_env.step(action)
        episode_reward += rewards[0]
        terminated = dones[0]
    #test_rewards.append(episode_reward)
    print(f"Test Episode {episode + 1}: Reward: {episode_reward}, Speed: {infos[0]['current_speed']}, Time: {infos[0]['time_elapsed']}")
#print(test_merges)

total_tests = len(test_merges)
successful_tests = sum(test_merges)
unsuccessful_tests = total_tests - successful_tests

print(f"Total Tests: {total_tests}")
print(f"Successful Merges: {successful_tests} ({(successful_tests/total_tests)*100:.2f}%)")
print(f"Unsuccessful Merges: {unsuccessful_tests} ({(unsuccessful_tests/total_tests)*100:.2f}%)")

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

rewards_moving_avg = moving_average(test_rewards, window_size=100)

# Plot the moving average reward
plt.figure(figsize=(10, 5))
plt.plot(rewards_moving_avg, label='Reward Moving Average (100 episodes)')
plt.title('Moving Average Reward (last 100 episodes) vs Episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()
