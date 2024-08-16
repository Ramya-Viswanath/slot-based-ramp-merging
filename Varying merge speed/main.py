import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import MlpExtractor
import torch as th
from torch import nn
from highway_env.vehicle.controller import MDPVehicle
import pandas as pd
from stable_baselines3.common.policies import ActorCriticPolicy

episodes_rewards = []
successful_merges = []
unsuccessful_merges = []
initial_speeds = []
vehicle_speeds = []
target_speeds = []
target_times = []

class CustomMergeEnv(gym.Env):
    
    def __init__(self, render_mode='human'):
        super().__init__()
        self.env = gym.make("merge-v0", render_mode=render_mode)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        original_obs_space = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(original_obs_space + 1,), dtype=np.float32)
        self.simulation_time = 0
        self.initial_speed = 15  
        self.target_speed = 30
        self.merging_distance = 280 
        self.merging_time = 0.80
        self.speed_limit = 40
        self.acceleration_range = (0, 6)  
        self.deceleration_range = (-3, 0)  
        self.goal_position = np.array([280, 0])
        self.viewer = None

    def reset(self, seed=None, options=None):
        #self.target_speed = float(np.random.choice([30, 35],p=[0.5,0.5]))
        self.target_speed = float(np.random.uniform(27,35))
        #print(f"Chosen target speed: {self.target_speed:.2f} m/s")
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
        obs = np.append(obs, self.target_speed)
        obs = obs[:self.observation_space.shape[0]]  
        return obs, {}

    def step(self, action):
        ego_vehicle = self.env.unwrapped.vehicle
        previous_speed = ego_vehicle.speed
        #print("step",self.target_speed)
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
            episodes_rewards.append(self.episode_reward)
            initial_speeds.append(self.initial_speed)
            vehicle_speeds.append(ego_vehicle.speed)
            target_speeds.append(self.target_speed)
            target_times.append(info['time_elapsed'])
            print(f"Episode ended. Initial Speed: {self.initial_speed:.2f} m/s, Merging Speed: {ego_vehicle.speed:.2f} m/s")
            print(f"Time Taken: {info['time_elapsed']:.2f} seconds, Reward: {self.episode_reward:.2f}")
            print(f"Target Speed: {self.target_speed:.2f} m/s, Merge: {'Perfect' if info['successful_merge'] else 'Imperfect'}")

            if info['successful_merge']:
                successful_merges.append(1)
                unsuccessful_merges.append(0)
            else:
                successful_merges.append(0)
                unsuccessful_merges.append(1)

        obs = np.append(obs, self.target_speed)
        obs = obs[:self.observation_space.shape[0]]
        return obs, self.episode_reward, terminated, truncated, info

    def calculate_reward(self, action, info, ego_vehicle):
        
        reward = 0

        if ego_vehicle.crashed:
            reward -= 10

        time_penalty = ((self.merging_time - info['time_elapsed'])/self.merging_time)**2
        speed_penalty = (abs(self.target_speed - ego_vehicle.speed)/self.target_speed)
        #print(self.target_speed)
        if ego_vehicle.position[0] > 230 :
             reward -= abs(self.target_speed - ego_vehicle.speed)/(self.target_speed)

            
        if ego_vehicle.position[0] >= self.goal_position[0]:
            if self.target_speed - 0.5 <= ego_vehicle.speed <= self.target_speed + 0.5 and self.merging_time - 0.05 <= info['time_elapsed'] <= self.merging_time + 0.05:
                print("Perfect Merging")
                reward += 10
                info['successful_merge'] = True
            else:
                reward -= 2
                print("Imperfect Merging")
                if ego_vehicle.speed < self.target_speed - 0.5 or ego_vehicle.speed > self.target_speed + 0.5:
                    reward -=  speed_penalty 
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
        if 'successful_merge' in self.locals['infos'][0]:
            if self.locals['infos'][0]['successful_merge']:
                self.successful_merges.append(1)
                self.unsuccessful_merges.append(0)
            else:
                self.successful_merges.append(0)
                self.unsuccessful_merges.append(1)
        return True

learning_rate = 3e-4
discount_factor = 0.95
net_arch = [64,64]  

def train_and_save_models():
    global episodes_rewards, successful_merges, unsuccessful_merges, initial_speeds, vehicle_speeds, target_speeds, target_times
    episodes_rewards = []
    successful_merges = []
    unsuccessful_merges = []
    initial_speeds = []
    vehicle_speeds = []
    target_speeds = []
    target_times = []
    
    env = make_vec_env(lambda: Monitor(CustomMergeEnv(render_mode='rgb_array'), info_keywords=('position', 'time_elapsed', 'current_speed')), n_envs=1)
    
    policy_kwargs = dict(
        net_arch=net_arch,
    )
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=discount_factor, n_steps=256, batch_size=64, n_epochs=10, policy_kwargs=policy_kwargs)

    total_timesteps = 400000
    callback = RenderAndLogCallback(total_timesteps=total_timesteps, render_timesteps=100)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("model_1.zip")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    training_rewards = episodes_rewards.copy()
    np.save("training_rewards_varying_initial_speed.npy", training_rewards)
    np.save("initial_speeds_varying_initial_speed.npy", initial_speeds)
    np.save("vehicle_speeds_varying_initial_speed.npy", vehicle_speeds)
    np.save("target_speeds_varying_initial_speed.npy", target_speeds)
    np.save("target_times_varying_initial_speed.npy", target_times)
    return training_rewards, mean_reward, std_reward

rewards, mean_reward, std_reward = train_and_save_models()
plt.figure(figsize=(15, 10))
moving_avg_rewards = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
initial_speeds = np.load("initial_speeds_varying_initial_speed.npy")
vehicle_speeds = np.load("vehicle_speeds_varying_initial_speed.npy")
target_speeds = np.load("target_speeds_varying_initial_speed.npy")
target_times = np.load("target_times_varying_initial_speed.npy")

plt.plot(moving_avg_rewards, label='Moving Average Reward', color='red')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Moving Average Reward vs Episodes (Initial speed Fixed at 20 m/s)')
plt.legend()
plt.savefig("training_rewards_fixed_initial_speed.png")
plt.show()

# Print results
successful_episodes = sum(successful_merges)
unsuccessful_episodes = sum(unsuccessful_merges)
total_episodes = successful_episodes + unsuccessful_episodes

print(f"Total Episodes: {total_episodes}")
print(f"Successful Merges: {successful_episodes} ({(successful_episodes / total_episodes) * 100:.2f}%)")
print(f"Unsuccessful Merges: {unsuccessful_episodes} ({(unsuccessful_episodes / total_episodes) * 100:.2f}%)")
