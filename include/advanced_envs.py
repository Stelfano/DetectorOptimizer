from scipy.stats import Uniform, gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from include.misc import simulate_tracks, eval_sim, draw_detector
from gymnasium import spaces


class banditOptimizer():
    '''
    Bandit optimizer is a simple Continuous multi armed bandit sub-environment designed to dynamically optimize
    the resolution parameters of the detector.
    
    It employs a "sliding window" approach by keeping only the WINDOW_SIZE last reward and parameters to have a distribution
    of resolution and rewards, as the iterative costructive process of the RL agent improves so do the rewards shifting the distribution
    to the optimal one
    '''
    
    def __init__(self, window_size, eps=0.1):
        self.window_size = window_size
        self.eps = eps                                   
        self.lo = np.array([0.001, 0.001, 0.001, 2])       
        self.hi = np.array([10.0, 10.0, 100.0, 20])
        self.kde = None

        self.samplers = {'time_sampler': Uniform(a=0.001, b=10), 'spatial_sampler': Uniform(a=0.001, b=10),
                         'margin_sampler': Uniform(a=0, b=100), 'N_sampler': Uniform(a=2, b=20)}
        self.parameters = {'time_params': np.array([]), 'spatial_params': np.array([]),
                           'tof_margin_params': np.array([]), 'N_params': np.array([])}
        self.losses = {'distances': np.array([]), 'costs': np.array([]), 'rewards': np.array([])}
        self.running_counter = 0

    def propose(self):
        if self.kde is not None and np.random.rand() > self.eps:
            for _ in range(5):
                cand = self.kde.resample(256).T              # batch economico dalla KDE
                cand[:, 3] = np.round(cand[:, 3])
                ok = np.all((cand >= self.lo) & (cand <= self.hi), axis=1)
                if ok.any():
                    return cand[ok][0]                       # primo campione valido: KDE troncata ai limiti
        s = {k: v.sample() for k, v in self.samplers.items()}
        return np.array([s['time_sampler'], s['spatial_sampler'],
                         s['margin_sampler'], round(s['N_sampler'])], dtype=float)

    def update(self, params, reward, cost=np.nan, distance=np.nan):
        time_resolution, spatial_resolution, tof_margin, n_planes = params

        if(len(self.parameters['time_params']) >= 100):
            self.parameters = {k: np.delete(v, 0) for k, v in self.parameters.items()}
            self.losses = {k: np.delete(v, 0) for k, v in self.losses.items()}

        self.parameters['time_params'] = np.append(self.parameters['time_params'], time_resolution)
        self.parameters['spatial_params'] = np.append(self.parameters['spatial_params'], spatial_resolution)
        self.parameters['tof_margin_params'] = np.append(self.parameters['tof_margin_params'], tof_margin)
        self.parameters['N_params'] = np.append(self.parameters['N_params'].astype(np.int32), round(n_planes))
        self.losses['distances'] = np.append(self.losses['distances'], distance)
        self.losses['costs'] = np.append(self.losses['costs'], cost)
        self.losses['rewards'] = np.append(self.losses['rewards'], reward)
        self.running_counter += 1

        if self.running_counter >= self.window_size:
            kde_data = np.vstack([self.parameters['time_params'], self.parameters['spatial_params'],
                                  self.parameters['tof_margin_params'], self.parameters['N_params']])
            T = self.losses['rewards'][self.losses['rewards'] >= np.quantile(self.losses['rewards'], 0.8)].std()
            w = np.exp((self.losses['rewards'] - self.losses['rewards'].max()) / T)
            try:
                self.kde = gaussian_kde(kde_data, weights=w)
                self.reward_dist = gaussian_kde(self.losses['rewards'])
            except np.linalg.LinAlgError:                    # NUOVO: covarianza singolare (es. N tutto uguale)
                pass
            self.running_counter = 0



MAX_PLANES = 20
class JointDetectorOptimizer(gym.Env):
    def __init__(self, n_tracks=200, spatial_res=0.1, time_res=0.3, tof_margin=50.0, n_planes = 5 , debug = False):
        self.bandit = banditOptimizer(100, eps = 0.2)
        self.debug = debug 
        self.n_tracks = n_tracks
        self.spatial_res = spatial_res
        self.time_res = time_res
        self.tof_margin = tof_margin
        self.n_planes = n_planes
        self.max_planes = MAX_PLANES

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = self.max_planes + 6
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)

        self.detector_state = np.array([], dtype=np.float32)
        self.global_seed = 42
        self.placement_range = (50.0, 1000.0)

    def _obs_and_metrics(self):
        n_planes = self.n_planes

        delta_t_meas = np.array([0])
        delta_t_true = np.array([0])
        config = [self.time_res, self.spatial_res, self.tof_margin, self.n_planes]
        detector_trimmed = self.detector_state[:self.max_planes]
        padded = np.pad(detector_trimmed, (0, self.max_planes - len(detector_trimmed)), constant_values=0)
        obs = np.concatenate([padded, delta_t_meas, delta_t_true, config]).astype(np.float32)

        if n_planes < 2:
            return obs, 0, 0, [], [], []

        x, y, z, _, _, _, delta_t_meas, delta_t_true, hit_valid = simulate_tracks(
            n_planes=n_planes, spatial_resolution=self.spatial_res,
            time_resolution=self.time_res, tof_margin=self.tof_margin,
            z_planes=self.detector_state.copy(), n_tracks=self.n_tracks,
            seed=self.global_seed,
        )

        return obs, delta_t_meas, delta_t_true, x, y, hit_valid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.detector_state = np.array([], dtype=np.float32)
        obs, _, _, _, _, _= self._obs_and_metrics()
        return obs, {}

    def calculate_reward(self, delta_t_meas, delta_t_true, hit_valid):
        if(len(self.detector_state) < 2):
            return 0, 0
        
        cost = self.n_planes * 10  + 1/self.spatial_res + self.tof_margin * 1.5 + 1/self.time_res**1.5

        rewards = ([
            1/1 - abs(delta_t_meas[i] - delta_t_true[i]) if hit_valid[i].all() else -0.5
            for i in range(delta_t_meas.shape[0])
        ])
        rewards = np.array(rewards).sum()
        return rewards, cost
                

    def step(self, action):
        plane_position = self.placement_range[0] + (action + 1.0) / 2.0 * (self.placement_range[1] - self.placement_range[0])
        plane_position = np.clip(plane_position, self.placement_range[0], self.placement_range[1])
        self.reward = 0.0

        terminated = False
        truncated = False

        if(self.detector_state.size == self.n_planes):
            terminated = True

        if plane_position not in self.detector_state:
            self.detector_state = np.sort(np.append(self.detector_state, plane_position))

        obs, delta_t_meas, delta_t_true, x_hits, y_hits, hit_valid = self._obs_and_metrics()
        self.reward, self.cost = self.calculate_reward(delta_t_meas, delta_t_true, hit_valid)
        if terminated :
            self.global_seed += 1
            if self.debug:
                print(f"Terminated: {terminated}, Detector state: {self.detector_state}, and reward: {self.reward}, cost: {self.cost}") 
            self.end_of_episode(self.reward, self.cost)
        return obs, float(self.reward), terminated, truncated, {}

    def end_of_episode(self, reward, cost):
        self.bandit.update(((self.time_res, self.spatial_res, self.tof_margin, self.n_planes)), reward=float(reward), cost=float(cost))
        new_config = self.bandit.propose()
        print("New config proposed by the bandit at:", new_config)
        self.time_res, self.spatial_res, self.tof_margin, self.n_planes = new_config[0], new_config[1], new_config[2], int(new_config[3])
            
    def render(self):
        obs, _, _, x_hits, y_hits, _ = self._obs_and_metrics()
        reward, cost = self.calculate_reward(self.detector_state, obs[1], obs[2])
        if len(self.detector_state) < 2:
            return
        draw_detector(n_planes=len(self.detector_state), z_planes=self.detector_state, plane_size=100.0, tof_margin=self.tof_margin, x_hits=x_hits, y_hits=y_hits, n_tracks_show=self.n_tracks, 
                      label=f"Detector 3D - Reward: {reward:.2f}, Planes: {len(self.detector_state)}\ndist theta: {obs[1]:.2f}, dist beta: {obs[2]:.2f}\ncost: {cost:.2f}, \nSpatial_res {self.spatial_res:.2f}, \nTof_margin {self.tof_margin:.2f}, \nTime_res {self.time_res:.2f}")