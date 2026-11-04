from scipy.stats import Uniform, gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from include.misc import simulate_tracks, eval_sim
from include.envs import SimplePotentialDetector


class banditOptimizer():
    def __init__(self, window_size, eps=0.1):
        self.window_size = window_size
        self.eps = eps                                   # NUOVO: quota di esplorazione uniforme
        self.lo = np.array([0.001, 0.001, 0.0, 2])       # NUOVO: limiti, gli stessi dei sampler
        self.hi = np.array([10.0, 10.0, 100.0, 20])
        self.kde = None

        # Samplers, invariati
        self.samplers = {'time_sampler': Uniform(a=0.001, b=10), 'spatial_sampler': Uniform(a=0.001, b=10),
                         'margin_sampler': Uniform(a=0, b=100), 'N_sampler': Uniform(a=2, b=20)}
        self.parameters = {'time_params': np.array([]), 'spatial_params': np.array([]),
                           'tof_margin_params': np.array([]), 'N_params': np.array([])}
        self.losses = {'distances': np.array([]), 'costs': np.array([]), 'rewards': np.array([])}
        self.running_counter = 0

    def propose(self):
        """Configurazione per il prossimo episodio: [time, spatial, tof_margin, N]."""
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


class BanditDetectorEnv(SimplePotentialDetector):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.bandit = banditOptimizer(100)
        self.current_cfg = None
        self.n_cap = self.max_planes
        self.last_cost = np.nan

    def calculate_reward(self, *args, **kwargs):              # memorizza il costo per passarlo al bandit
        reward, cost = super().calculate_reward(*args, **kwargs)
        self.last_cost = cost
        return reward, cost

    def reset(self, seed=None, options=None):
        self.current_cfg = self.bandit.propose()
        self.time_res, self.spatial_res, self.tof_margin = map(float, self.current_cfg[:3])
        self.n_cap = min(int(self.current_cfg[3]), self.max_planes)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, r, terminated, truncated, info = super().step(action)

        if not (terminated or truncated) and len(self.detector_state) >= self.n_cap:
            truncated = True                                  # tetto di piani scelto dal bandit
            self.global_seed += 1                             # come fa il parent a fine episodio

        if (terminated or truncated) and self.current_cfg is not None:
            reward = float(self.reward)                       # reward grezzo del detector finale (senza il /10)
            self.bandit.update(self.current_cfg, reward, cost=float(self.last_cost))
            info = {**info, 'config': self.current_cfg.copy(), 'config_reward': reward}
            self.current_cfg = None                           # un solo update per episodio
        return obs, r, terminated, truncated, info