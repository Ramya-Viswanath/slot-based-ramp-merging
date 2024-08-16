import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from highway_env.vehicle.controller import MDPVehicle
import warnings
import time

episodes_rewards = []

#Custom Environment for ramp merging
class CustomMergeEnv(gym.Env):
    def __init__(self, render_mode='human'):
        super().__init__()
        self.env = gym.make("merge-v0", render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = self.env.observation_space
        self.simulation_time = 0
        self.initial_speed = 10  
        self.target_speed = 28.5 
        self.merging_distance = 290 
        self.merging_time = 0.8  
        self.speed_limit = 35 
        self.acceleration_range = (0, 6)  
        self.deceleration_range = (-3, 0)  
        self.goal_position = np.array([290,0])

    #Resetting the environment 
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
            route=[("j", "k", 0), ("k", "b", 0)])  
        road.vehicles.append(ego_vehicle)
        self.env.unwrapped.vehicle = ego_vehicle
        self.simulation_time = 0
        return self.env.unwrapped.observation_type.observe(), {}

    #Observation Space
    def step(self, action):
        ego_vehicle = self.env.unwrapped.vehicle
        previous_speed = ego_vehicle.speed

        if ego_vehicle.speed > self.speed_limit: 
            ego_vehicle.speed = self.speed_limit
        if action == 1:  
            acceleration = np.random.uniform(*self.acceleration_range)
            ego_vehicle.target_speed = min(self.speed_limit, ego_vehicle.speed + acceleration)
        elif action == 2: 
            deceleration = np.random.uniform(*self.deceleration_range)
            ego_vehicle.target_speed = max(0.1, ego_vehicle.speed + deceleration)  # Ensure target_speed is not zero
        else:
            ego_vehicle.target_speed = ego_vehicle.speed 

        ego_vehicle.step(1 / self.env.unwrapped.config["simulation_frequency"])  # Update vehicle state
        obs, reward, terminated, truncated, info = self.env.step(0)
        self.simulation_time += 1 / self.env.unwrapped.config["simulation_frequency"]
        info['position'] = 'motorway' if ego_vehicle.position[0] >= self.merging_distance else 'ramp'
        info['previous_speed'] = previous_speed
        info['time_elapsed'] = self.simulation_time
        info['current_speed'] = ego_vehicle.speed
        custom_reward = self.calculate_reward(action, info,ego_vehicle)

        if ego_vehicle.position[0] >= self.goal_position[0]:  
            terminated = True
        if terminated or truncated:
            print(f"Episode ended. Merging speed: {ego_vehicle.speed} m/s, Time taken: {info['time_elapsed']} seconds, Reward: {custom_reward}")
            episodes_rewards.append(custom_reward)
        return obs, custom_reward, terminated, truncated, info

    #Reward system
    def calculate_reward(self, action, info, ego_vehicle):
        reward = 0

        if ego_vehicle.crashed:
            reward -= 10
        if ego_vehicle.position[0] >= self.goal_position[0]:
            if self.target_speed - 0.5 <= ego_vehicle.speed <= self.target_speed + 0.5 and self.merging_time - 0.1 <= info['time_elapsed'] <= self.merging_time + 0.1:
                reward += 10
            else:
                if ego_vehicle.speed < self.target_speed - 0.5 or ego_vehicle.speed > self.target_speed + 0.5:
                    reward -= 2
                if info['time_elapsed'] > self.merging_time + 0.1 or info['time_elapsed'] < self.merging_time - 0.1:
                    reward -= 2

        if ego_vehicle.speed > self.speed_limit or ego_vehicle.speed <= 10:
            reward -= 1

        if info['time_elapsed'] > self.merging_time + 0.1 and ego_vehicle.position[0] < self.goal_position[0]:
            reward -= 1
        if action == 1 and ego_vehicle.speed - info['previous_speed'] > self.acceleration_range[1]:
            reward -= 1
        if action == 2 and ego_vehicle.speed - info['previous_speed'] < self.deceleration_range[0]:
            reward -= 1
        return reward

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()

#For rendering
class RenderAndLogCallback(BaseCallback):
    def __init__(self, total_timesteps, render_timesteps=100, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.render_timesteps = render_timesteps
        self.current_timestep = 0

    def _on_step(self) -> bool:
        self.current_timestep += 1
        if self.current_timestep >= self.total_timesteps - self.render_timesteps:
            for i in range(len(self.training_env.envs)):
                self.training_env.envs[i].render()
                time.sleep(0.3)
        return True

#PPO SB3 Creation
env = CustomMergeEnv(render_mode='rgb_array') 
env = Monitor(env, info_keywords=('position', 'time_elapsed', 'current_speed'))
vec_env = make_vec_env(lambda: Monitor(CustomMergeEnv(render_mode='rgb_array'), info_keywords=('position', 'time_elapsed', 'current_speed')), n_envs=1)
model = PPO('MlpPolicy', vec_env, verbose=1, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10)
timesteps_per_episode = 100
episodes = 1000  
total_timesteps = timesteps_per_episode * episodes 
callback = RenderAndLogCallback(total_timesteps=total_timesteps, render_timesteps=100)

model.learn(total_timesteps=total_timesteps,callback=callback)
model.save("scenario1_model")
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

if len(episodes_rewards) > 0:
    plt.figure(figsize=(10, 5))

    window_size = 50
    moving_avg_rewards = []
    for i in range(len(episodes_rewards)):
        if i < window_size:
            moving_avg_rewards.append(np.mean(episodes_rewards[:i+1]))
        else:
            moving_avg_rewards.append(np.mean(episodes_rewards[i-window_size+1:i+1]))

    plt.plot(range(len(episodes_rewards)), moving_avg_rewards, label='Moving Average (50 episodes)', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episodes')
    plt.legend()
    plt.show()
else:
    print("No rewards to plot.")
