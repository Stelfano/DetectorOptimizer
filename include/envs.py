from include.misc import simulate_tracks, eval_sim, draw_detector, reward_calculator
import gymnasium as gym
from gymnasium import spaces
import numpy as np

MAX_PLANES = 20  


class SimpleRoTDetector(gym.Env):
    def __init__(self, n_tracks=200, spatial_res=0.1, time_res=0.3, tof_margin=50.0, debug = False, reward_mode = 'length'):
        self.debug = debug 
        super().__init__()
        self.n_tracks = n_tracks
        self.spatial_res = spatial_res
        self.time_res = time_res
        self.tof_margin = tof_margin
        self.max_planes = MAX_PLANES
        self.reward_calculator = reward_calculator(reward_mode)
        

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dim = self.max_planes + 2
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

        self.detector_state = np.array([], dtype=np.float32)
        self.global_seed = 42
        self.placement_range = (50.0, 1000.0)

    def _obs_and_metrics(self):
        n_planes = len(self.detector_state)
        if n_planes < 2:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 999.0, 999.0, [], []

        x, y, z, theta, theta_deg, beta, dt, hit_valid = simulate_tracks(
            n_planes=n_planes, spatial_resolution=self.spatial_res,
            time_resolution=self.time_res, tof_margin=self.tof_margin,
            z_planes=self.detector_state.copy(), n_tracks=self.n_tracks,
            seed=self.global_seed,
        )
        self.global_seed += 1
        ev = eval_sim(x, y, z, dt, hit_valid.astype(bool), n_planes)
        theta_dist = float(np.linalg.norm(ev["theta_measured"] - theta_deg))
        beta_dist = float(np.linalg.norm(ev["beta_measured"] - beta))

        padded = np.zeros(self.max_planes, dtype=np.float32)
        padded[:n_planes] = self.detector_state
        obs = np.concatenate([padded, [theta_dist, beta_dist]]).astype(np.float32)

        #normalize the observation vector to have a unit norm, which can help with training stability in reinforcement learning.
        obs /= np.linalg.norm(obs) + 1e-8

        return obs, theta_dist, beta_dist, x, y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.detector_state = np.array([], dtype=np.float32)
        obs, _, _, _, _= self._obs_and_metrics()
        return obs, {}

    def calculate_reward(self, theta_dist, beta_dist):
        return self.reward_calculator.forward(self.detector_state, beta_dist, theta_dist)

    def step(self, action):
        plane_position = self.placement_range[0] + (float(action[0]) + 1.0) / 2.0 * (self.placement_range[1] - self.placement_range[0])
        plane_position = float(np.clip(plane_position, self.placement_range[0], self.placement_range[1]))
        stop_signal = float(action[1])
        reward = 0.0

        should_stop = stop_signal > 0.5 and len(self.detector_state) > 1
        terminated = False
        truncated = False

        if(self.detector_state.size == self.max_planes):
            truncated = True

        if should_stop:
            terminated = True
        else:
            if plane_position not in self.detector_state and not truncated:
                self.detector_state = np.sort(np.append(self.detector_state, plane_position))

        obs, theta_dist, beta_dist, x_hits, y_hits = self._obs_and_metrics()
        if terminated or truncated:
            reward, cost = self.calculate_reward(theta_dist, beta_dist)
            if self.debug:
                print(f"Terminated: {terminated}, Truncated: {truncated}, Detector state: {self.detector_state}, and stop_signal: {stop_signal}, and reward: {reward}, cost: {cost}") 
        return obs, float(reward)/10, terminated, truncated, {}

    
    def render(self):
        obs, _, _, x_hits, y_hits = self._obs_and_metrics()
        reward, cost = self.calculate_reward(obs[1], obs[2])
        if len(self.detector_state) < 2:
            return
        draw_detector(n_planes=len(self.detector_state), z_planes=self.detector_state, plane_size=100.0, tof_margin=50, x_hits=x_hits, y_hits=y_hits, n_tracks_show=self.n_tracks, label=f"Detector 3D - Reward: {reward:.2f}, Planes: {len(self.detector_state)}\ndist theta: {obs[1]:.2f}, dist beta: {obs[2]:.2f}\ncost: {cost:.2f}")


class SimplePotentialDetector(SimpleRoTDetector):
    def __init__(self, n_tracks=200, spatial_res=0.1, time_res=0.3, tof_margin=50.0):
        super().__init__(n_tracks=n_tracks, spatial_res=spatial_res, time_res=time_res, tof_margin=tof_margin)
        
    def step(self, action):
        plane_position = self.placement_range[0] + (float(action[0]) + 1.0) / 2.0 * (self.placement_range[1] - self.placement_range[0])
        plane_position = float(np.clip(plane_position, self.placement_range[0], self.placement_range[1]))
        stop_signal = float(action[1])
        reward = 0.0
        
        should_stop = stop_signal > 0.5 and len(self.detector_state) > 1
        terminated = False
        truncated = False
       
        if(self.detector_state.size == self.max_planes):
            truncated = True
        
        if should_stop:
            terminated = True
        else:
            if plane_position not in self.detector_state and not truncated:
                self.detector_state = np.sort(np.append(self.detector_state, plane_position))
        
        obs, theta_dist, beta_dist, x_hits, y_hits = self._obs_and_metrics()
        if(len(self.detector_state) > 1):
            reward, cost = self.calculate_reward(theta_dist, beta_dist)
            if self.debug:
                print(f"Terminated: {terminated}, Truncated: {truncated}, Detector state: {self.detector_state}, and stop_signal: {stop_signal}, and reward: {reward}, cost: {cost}") 
        return obs, float(reward)/10, terminated, truncated, {}
        