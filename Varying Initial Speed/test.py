import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from highway_env.vehicle.controller import MDPVehicle

test_rewards = []
test_successful_merges = []
test_unsuccessful_merges = []
test_initial_speeds = []
test_vehicle_speeds = []
test_target_speeds = []
test_target_times = []

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
        self.initial_speed = float(np.random.uniform(10, 30))  # Random initial speed between 10 and 30 m/s
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
        elif acceleration < 0:
            ego_vehicle.target_speed = max(0.1, ego_vehicle.speed + acceleration)
        else:
            ego_vehicle.target_speed = ego_vehicle.speed

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
            test_rewards.append(self.episode_reward)
            test_initial_speeds.append(self.initial_speed)
            test_vehicle_speeds.append(ego_vehicle.speed)
            test_target_speeds.append(self.target_speed)
            test_target_times.append(info['time_elapsed'])
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
                    reward -= time_penalty
                if info['time_elapsed'] > self.merging_time + 0.05 or info['time_elapsed'] < self.merging_time - 0.05:
                    reward -= speed_penalty
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

test_episodes = 5000
model = PPO.load("ppo_merge_varying_initial_speed.zip")
test_vec_env = make_vec_env(lambda: Monitor(CustomMergeEnv(render_mode='rgb_array'), info_keywords=('position', 'time_elapsed', 'current_speed')), n_envs=1)

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
    test_initial_speeds.append(test_vec_env.get_attr('initial_speed', 0))
    test_vehicle_speeds.append(test_vec_env.get_attr('env', 0)[0].unwrapped.env.vehicle.speed)

    test_target_speeds.append(test_vec_env.get_attr('target_speed', 0))
    test_target_times.append(infos[0]['time_elapsed'])
    if successful_merge:
        test_successful_merges.append(1)
    else:
        test_unsuccessful_merges.append(1)
   # print(f"Test Episode {episode + 1} ended. Initial Speed: {test_initial_speeds[-1]:.2f} m/s, Merging Speed: {test_vehicle_speeds[-1]:.2f} m/s, Time Taken: {test_target_times[-1]:.2f} seconds, Reward: {episode_reward:.2f}, Merge: {'Perfect' if successful_merge else 'Imperfect'}")

total_test_episodes = len(test_successful_merges) + len(test_unsuccessful_merges)
successful_test_episodes = sum(test_successful_merges)
unsuccessful_test_episodes = sum(test_unsuccessful_merges)

print(f"Testing Episodes: {total_test_episodes}")
print(f"Successful Testing Merges: {successful_test_episodes} ({(successful_test_episodes / total_test_episodes) * 100:.2f}%)")
print(f"Unsuccessful Testing Merges: {unsuccessful_test_episodes} ({(unsuccessful_test_episodes / total_test_episodes) * 100:.2f}%)")

plt.figure(figsize=(15, 10))
moving_avg_test_rewards = [np.mean(test_rewards[max(0, i-99):i+1]) for i in range(len(test_rewards))]
plt.plot(moving_avg_test_rewards, label='Moving Average Reward',color='red')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Moving Average Reward (last 100 episodes) vs Episode (Initial speed Variation)')
plt.legend()
plt.savefig("test_rewards_varying_initial_speed.png")
plt.show()

