from include.misc import simulate_tracks, eval_sim, draw_detector, reward_calculator
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import warnings
from skopt import gp_minimize

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
        ev = eval_sim(x, y, z, dt, hit_valid.astype(bool), n_planes)
        theta_dist = float(np.linalg.norm(ev["theta_measured"] - theta_deg))
        beta_dist = float(np.linalg.norm(ev["beta_measured"] - beta))

        padded = np.zeros(self.max_planes, dtype=np.float32)
        padded[:n_planes] = self.detector_state
        obs = np.concatenate([padded, [theta_dist, beta_dist]]).astype(np.float32)

        #normalize the observation vector to have a unit norm, which can help with training stability in reinforcement learning.
        return obs, theta_dist, beta_dist, x, y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.detector_state = np.array([], dtype=np.float32)
        obs, _, _, _, _= self._obs_and_metrics()
        return obs, {}

    def calculate_reward(self, detector_state, theta_dist, beta_dist, spatial_res=None, tof_margin=None, time_res=None):
        return self.reward_calculator.forward(detector_state, beta_dist, theta_dist, self.spatial_res, self.tof_margin, self.time_res)

    def step(self, action):
        plane_position = self.placement_range[0] + (float(action[0]) + 1.0) / 2.0 * (self.placement_range[1] - self.placement_range[0])
        plane_position = float(np.clip(plane_position, self.placement_range[0], self.placement_range[1]))
        stop_signal = float(action[1])
        self.reward = 0.0

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
            self.global_seed += 1
            self.end_of_episode()
            self.reward, cost = self.calculate_reward(self.detector_state, theta_dist, beta_dist)
            if self.debug:
                print(f"Terminated: {terminated}, Truncated: {truncated}, Detector state: {self.detector_state}, and stop_signal: {stop_signal}, and reward: {self.reward}, cost: {cost}") 
        return obs, float(self.reward)/10, terminated, truncated, {}

    def end_of_episode(self):
        return
    
    def render(self):
        obs, _, _, x_hits, y_hits = self._obs_and_metrics()
        reward, cost = self.calculate_reward(self.detector_state, obs[1], obs[2])
        if len(self.detector_state) < 2:
            return
        draw_detector(n_planes=len(self.detector_state), z_planes=self.detector_state, plane_size=100.0, tof_margin=self.tof_margin, x_hits=x_hits, y_hits=y_hits, n_tracks_show=self.n_tracks, 
                      label=f"Detector 3D - Reward: {reward:.2f}, Planes: {len(self.detector_state)}\ndist theta: {obs[1]:.2f}, dist beta: {obs[2]:.2f}\ncost: {cost:.2f}, \nSpatial_res {self.spatial_res:.2f}, \nTof_margin {self.tof_margin:.2f}, \nTime_res {self.time_res:.2f}")


class SimplePotentialDetector(SimpleRoTDetector):
    def __init__(self, n_tracks=200, spatial_res=0.1, time_res=0.3, tof_margin=50.0, debug = False, reward_mode = 'length'):
        super().__init__(n_tracks=n_tracks, spatial_res=spatial_res, time_res=time_res, tof_margin=tof_margin, debug=debug, reward_mode=reward_mode)
        
    def step(self, action):
        plane_position = self.placement_range[0] + (float(action[0]) + 1.0) / 2.0 * (self.placement_range[1] - self.placement_range[0])
        plane_position = float(np.clip(plane_position, self.placement_range[0], self.placement_range[1]))
        stop_signal = float(action[1])
        self.reward = 0.0
        
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
            self.reward, cost = self.calculate_reward(self.detector_state, theta_dist, beta_dist, self.spatial_res, self.tof_margin, self.time_res)
            if self.debug:
                print(f"Terminated: {terminated}, Truncated: {truncated}, Detector state: {self.detector_state}, and stop_signal: {stop_signal}, and reward: {self.reward}, cost: {cost}") 

            if terminated or truncated:
                self.global_seed += 1
        return obs, float(self.reward)/10, terminated, truncated, {}
        

class classicalOptimizingDetector(SimpleRoTDetector):
    def __init__(self, n_tracks=200, spatial_res=0.1, time_res=0.3, tof_margin=50, debug=False, reward_mode='length'):
        super().__init__(n_tracks, spatial_res, time_res, tof_margin, debug, reward_mode)
        self.reward_calculator = reward_calculator(mode='length_params')


    def end_of_episode(self):
        def f(x):
            self.spatial_res = x[0]
            self.tof_margin = x[1]
            self.time_res = x[2]
            _, theta, beta, _, _ = self._obs_and_metrics()
            return - self.reward_calculator.forward(self.detector_state, beta, theta, x[0], x[1], x[2])[0]

        res = gp_minimize(f, [(0.01, 10), (1, 100), (0.01, 10)], n_calls=20)
        
        self.spatial_res = res.x[0]
        self.tof_margin = res.x[1]
        self.time_res = res.x[2]