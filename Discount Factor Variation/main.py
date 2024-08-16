import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from highway_env.vehicle.controller import MDPVehicle
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd

colors = ['red','orange','green']

episodes_rewards = []
successful_merges = []
unsuccessful_merges = []

test_rewards = []
test_successful_merges = []
test_unsuccessful_merges = []

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
            episodes_rewards.append(self.episode_reward)
            if info['successful_merge']:
                successful_merges.append(1)
                unsuccessful_merges.append(0)
            else:
                successful_merges.append(0)
                unsuccessful_merges.append(1)
        return obs, self.episode_reward, terminated, truncated, info

    def calculate_reward(self, action, info, ego_vehicle):
        reward = 0

        if ego_vehicle.crashed:
            reward -= 10

        time_penalty = ((self.merging_time - info['time_elapsed'])/self.merging_time)**2
        speed_penalty = ((self.target_speed - ego_vehicle.speed)/self.target_speed)**2

        if ego_vehicle.position[0] >= self.goal_position[0]:
            if self.target_speed - 0.5 <= ego_vehicle.speed <= self.target_speed + 0.5 and self.merging_time - 0.05 <= info['time_elapsed'] <= self.merging_time + 0.05:
                reward += 10
                info['successful_merge'] = True
            else:
                reward -= 2
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
        if 'successful_merge' in self.locals['infos'][0]:
            if self.locals['infos'][0]['successful_merge']:
                self.successful_merges.append(1)
                self.unsuccessful_merges.append(0)
            else:
                self.successful_merges.append(0)
                self.unsuccessful_merges.append(1)
        return True

learning_rate = 3e-4
discount_factors = [0.99, 0.95, 0.90]

def train_and_save_models(discount_factor):
    global episodes_rewards, successful_merges, unsuccessful_merges
    episodes_rewards = []
    successful_merges = []
    unsuccessful_merges = []
    
    env = make_vec_env(lambda: Monitor(CustomMergeEnv(render_mode='rgb_array'), info_keywords=('position', 'time_elapsed', 'current_speed')), n_envs=1)
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=discount_factor, n_steps=256, batch_size=64, n_epochs=10)
    total_timesteps = 400000
    callback = RenderAndLogCallback(total_timesteps=total_timesteps, render_timesteps=100)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(f"ppo_merge_gamma_{discount_factor}.zip")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    training_rewards = episodes_rewards.copy()
    np.save(f"training_rewards_gamma_{discount_factor}.npy", training_rewards)
    return training_rewards, mean_reward, std_reward

results = []
for gamma in discount_factors:
    rewards, mean_reward, std_reward = train_and_save_models(gamma)
    results.append((gamma, rewards, mean_reward, std_reward))

results_df = pd.DataFrame(results, columns=["Discount Factor", "Training Rewards", "Mean Reward", "Std Reward"])
results_df.to_csv("training_results_gamma.csv", index=False)
results_df = pd.read_csv("training_results_gamma.csv")

j=0
plt.figure(figsize=(15, 10))
for gamma in discount_factors:
    rewards = np.load(f"training_rewards_gamma_{gamma}.npy")
    moving_avg_rewards = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
    plt.plot(moving_avg_rewards, label=f'Gamma: {gamma}',color = colors[j])
    j+=1
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Moving Average (last 100 episodes) vs Episodes for Different Discount Factors')
plt.legend(title = 'Discount Factor')
plt.savefig("training_rewards_different_discount_factors.png")
plt.show()


test_episodes = 5000
test_results = []
for gamma in discount_factors:
    model = PPO.load(f"ppo_merge_gamma_{gamma}.zip")
    test_vec_env = make_vec_env(lambda: Monitor(CustomMergeEnv(render_mode='rgb_array'), info_keywords=('position', 'time_elapsed', 'current_speed')), n_envs=1)
    test_successful_merges = []
    test_unsuccessful_merges = []
    test_rewards = []
    for episode in range(test_episodes):
        obs = test_vec_env.reset()
        episode_reward = 0
        terminated = False
        successful_merge = False
        while not terminated:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = test_vec_env.step(action)
            episode_reward += rewards[0]
            terminated = dones[0]
            if 'successful_merge' in infos[0] and infos[0]['successful_merge']:
                successful_merge = True
        test_rewards.append(episode_reward)
        if successful_merge:
            test_successful_merges.append(1)
        else:
            test_unsuccessful_merges.append(1)
    total_test_episodes = test_episodes
    successful_test_episodes = sum(test_successful_merges)
    unsuccessful_test_episodes = sum(test_unsuccessful_merges)

    print(f"Discount Factor: {gamma}")
    print(f"Testing Episodes: {test_episodes}")
    print(f"Successful Testing Merges: {successful_test_episodes} ({(successful_test_episodes / total_test_episodes) * 100:.2f}%)")
    print(f"Unsuccessful Testing Merges: {unsuccessful_test_episodes} ({(unsuccessful_test_episodes / total_test_episodes) * 100:.2f}%)")
    print()
    test_results.append((gamma, test_rewards.copy(), successful_test_episodes, unsuccessful_test_episodes))
test_results_df = pd.DataFrame(test_results, columns=["Discount Factor", "Test Rewards", "Successful Merges", "Unsuccessful Merges"])
test_results_df.to_csv("test_results_gamma.csv", index=False)

j=0
plt.figure(figsize=(15, 10))
for gamma, test_rewards, _, _ in test_results:
    moving_avg_test_rewards = [np.mean(test_rewards[max(0, i-99):i+1]) for i in range(len(test_rewards))]
    plt.plot(moving_avg_test_rewards, label=f'{gamma}',color= colors[j])
    j+=1
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Moving Rewards (last 100 episodes) vs Episodes for Different Discount Factors')
plt.legend(title = 'Discount Factor')
plt.savefig("test_rewards_comparison_gamma.png")
plt.show()
