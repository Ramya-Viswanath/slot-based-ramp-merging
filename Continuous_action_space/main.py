#Training code for continuous action space for 3,50,000 timesteps.

import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from highway_env.vehicle.controller import MDPVehicle
import warnings
import time
import cv2

episodes_rewards = []
episodes_speeds = []
episodes_times = []
successful_merges = []
unsuccessful_merges = []

test_rewards = []
test_speeds = []
test_times = []
test_successful_merges = []
test_unsuccessful_merges = []

#Custom environment
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

    #Environment restes as the episode ends
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
        self.episode_reward = 0 
        return self.env.unwrapped.observation_type.observe(), {}

    #Observation space or state space
    def step(self, action):
        
        ego_vehicle = self.env.unwrapped.vehicle
        previous_speed = ego_vehicle.speed

        if ego_vehicle.speed > self.speed_limit: 
            ego_vehicle.speed = self.speed_limit
        #Defining the range for acceleration and decelaration
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

        #Checks if the episode is terminated
        if terminated or truncated:
            print(f"Episode ended. Merging speed: {ego_vehicle.speed:.2f} m/s, Time taken: {info['time_elapsed']:.2f} seconds, Reward: {self.episode_reward:.2f}")
            episodes_rewards.append(self.episode_reward)
            episodes_speeds.append(ego_vehicle.speed)
            episodes_times.append(info['time_elapsed'])
            if info['successful_merge']:
                successful_merges.append(1)
                unsuccessful_merges.append(0)
            else:
                successful_merges.append(0)
                unsuccessful_merges.append(1)
            print()
        return obs, self.episode_reward, terminated, truncated, info

    #Reward system
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

class RenderAndLogCallback(BaseCallback):
    
    def __init__(self, total_timesteps, render_timesteps=100, verbose=0):
        
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.render_timesteps = render_timesteps
        self.current_timestep = 0
        self.frames = []
        self.successful_merges = []
        self.unsuccessful_merges = []

    def _on_step(self) -> bool:
        
        self.current_timestep += 1
        if self.current_timestep >= self.total_timesteps - self.render_timesteps:
            for i in range(len(self.training_env.envs)):
                frame = self.training_env.envs[i].env.render()
                if frame is not None:
                    self.frames.append(frame)
                else:
                    print("Warning: Captured frame is None")
                time.sleep(0.03)

        if 'successful_merge' in self.locals['infos'][0]:
            
            if self.locals['infos'][0]['successful_merge']:
                self.successful_merges.append(1)
                self.unsuccessful_merges.append(0)
            else:
                self.successful_merges.append(0)
                self.unsuccessful_merges.append(1)

        return True

    def save_video(self, filepath="merge_continuous.mp4"):
        
        if len(self.frames) == 0:
            print("No frames to save.")
            return
        height, width, layers = self.frames[0].shape
        video = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        for frame in self.frames:
            if frame is not None:
                video.write(frame)
        video.release()

env = CustomMergeEnv(render_mode='human')  
env = Monitor(env, info_keywords=('position', 'time_elapsed', 'current_speed'))
vec_env = make_vec_env(lambda: Monitor(CustomMergeEnv(render_mode='rgb_array'), info_keywords=('position', 'time_elapsed', 'current_speed')), n_envs=1)
model = PPO('MlpPolicy', vec_env, verbose=1, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10, gamma=0.90)

total_timesteps = 350000
callback = RenderAndLogCallback(total_timesteps=total_timesteps, render_timesteps=100)

model.learn(total_timesteps=total_timesteps, callback=callback)
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
model.save("ramp_merging_model")

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
    plt.savefig("training_rewards.png")
    plt.show()
else:
    print("No rewards to plot.")

if len(episodes_speeds) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(episodes_speeds)), episodes_speeds, label='Vehicle Speed', color='blue')
    plt.axhline(y=env.target_speed, color='r', linestyle='--', label='Target Speed')
    plt.xlabel('Episode')
    plt.ylabel('Speed')
    plt.title('Vehicle Speed vs Episode')
    plt.legend()
    plt.savefig("training_speeds.png")
    plt.show()
else:
    print("No speeds to plot.")

if len(episodes_times) > 0:
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(episodes_times)), episodes_times, label='Merging Time', color='blue')
    plt.axhline(y=env.merging_time, color='r', linestyle='--', label='Target Merging Time')
    plt.xlabel('Episode')
    plt.ylabel('Time (seconds)')
    plt.title('Merging Time vs Episode')
    plt.legend()
    plt.savefig("training_times.png")
    plt.show()
else:
    print("No merging times to plot.")

