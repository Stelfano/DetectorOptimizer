import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import logging
import os


def simulate_tracks(
    n_planes, spatial_resolution, n_tracks=1000, plane_size=100.0,
    start_area_fraction=0.10, z_min=0.0, z_max=1000.0, z_planes=None,
    theta_max_deg=10.0, beta_min=0.5, beta_max=0.75, time_resolution=2.0,
    tof_margin=50.0, default_value=-999999.0, seed=42,
):
    """Simula tracce Monte Carlo attraverso i piani del detector (invariata rispetto all'originale)."""
    rng = np.random.default_rng(seed)
    c_mm_ns = 299.792458

    if z_planes is None or len(z_planes) == 0:
        z_planes = np.linspace(z_min, z_max, n_planes)
    z_planes = np.asarray(z_planes, dtype=np.float64)

    z_time_1 = z_min - tof_margin
    z_time_2 = z_max + tof_margin
    delta_z_tof = z_time_2 - z_time_1

    start_size = plane_size * np.sqrt(start_area_fraction)
    start_half_size = start_size / 2

    x0 = rng.uniform(-start_half_size, start_half_size, n_tracks)
    y0 = rng.uniform(-start_half_size, start_half_size, n_tracks)

    theta_max = np.deg2rad(theta_max_deg)
    phi = rng.uniform(0, 2 * np.pi, n_tracks)
    cos_theta = rng.uniform(np.cos(theta_max), 1, n_tracks)
    theta = np.arccos(cos_theta)
    theta_deg = np.rad2deg(theta)
    tan_theta = np.tan(theta)

    tx = tan_theta * np.cos(phi)
    ty = tan_theta * np.sin(phi)
    beta = rng.uniform(beta_min, beta_max, n_tracks)

    x_true = x0[:, None] + tx[:, None] * z_planes[None, :]
    y_true = y0[:, None] + ty[:, None] * z_planes[None, :]

    half_plane_size = plane_size / 2
    hit_valid = (np.abs(x_true) <= half_plane_size) & (np.abs(y_true) <= half_plane_size)

    x_meas = x_true + rng.normal(0, spatial_resolution, size=x_true.shape)
    y_meas = y_true + rng.normal(0, spatial_resolution, size=y_true.shape)
    x_meas[~hit_valid] = default_value
    y_meas[~hit_valid] = default_value

    path_length = delta_z_tof / np.cos(theta)
    delta_t_true = path_length / (beta * c_mm_ns)

    t1_meas = rng.normal(0, time_resolution, n_tracks)
    t2_meas = delta_t_true + rng.normal(0, time_resolution, n_tracks)
    delta_t_meas = t2_meas - t1_meas

    return x_meas, y_meas, z_planes, theta, theta_deg, beta, delta_t_meas, hit_valid


#Rimuovere questo step e fare in modo che la rete sia in grado di fare la stima di beta e theta direttamente dai dati misurati
def eval_sim(x_meas, y_meas, z_planes, delta_t_meas, hit_valid, n_planes):
    i_start, i_end = 0, n_planes - 1
    dx = x_meas[:, i_end] - x_meas[:, i_start]
    dy = y_meas[:, i_end] - y_meas[:, i_start]
    dz = z_planes[i_end] - z_planes[i_start]
    dr = np.sqrt(dx**2 + dy**2)
    path_length = np.sqrt(dr**2 + dz**2)

    c = 299.792458
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_measured = path_length / (c * delta_t_meas)

    theta_rad_est = np.arctan2(dr, dz)
    theta_deg_est = np.degrees(theta_rad_est)

    mask = hit_valid[:, i_start] & hit_valid[:, i_end]
    beta_measured = np.where(mask, beta_measured, -5.0)
    theta_deg_est = np.where(mask, theta_deg_est, -5.0)

    return {"beta_measured": beta_measured, "theta_measured": theta_deg_est}

    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_detector(
    n_planes,
    z_planes,
    plane_size=100.0,
    tof_margin=50.0,
    x_hits=None,
    y_hits=None,
    n_tracks_show=20,
    label = "Detector 3D"
):
    """
    Disegna il detector in 3D usando la coordinata fisica z
    come asse orizzontale del grafico.

    Mapping del plot:
        asse X grafico = z fisico
        asse Y grafico = x fisico
        asse Z grafico = y fisico
    """

    z_planes = z_planes[:n_planes]
    x_hits = x_hits[:, :n_planes]
    y_hits = y_hits[:, :n_planes]
    fig = plt.figure(figsize=(32, 10))
    ax = fig.add_subplot(111, projection="3d")

    half_size = plane_size / 2

    z_tracker_min = z_planes[0]
    z_tracker_max = z_planes[-1]

    z_time_1 = z_tracker_min - tof_margin
    z_time_2 = z_tracker_max + tof_margin

    def add_plane(z_phys, label, facecolor, alpha=0.25):
        """
        Piano fisico perpendicolare a z.
        Nel plot viene disegnato a X = z_phys,
        esteso lungo Y = x e Z = y.
        """

        vertices = [
            [
                (z_phys, -half_size, -half_size),
                (z_phys,  half_size, -half_size),
                (z_phys,  half_size,  half_size),
                (z_phys, -half_size,  half_size),
            ]
        ]

        plane = Poly3DCollection(
            vertices,
            facecolors=facecolor,
            edgecolors="black",
            linewidths=1.0,
            alpha=alpha
        )

        ax.add_collection3d(plane)

        ax.text(
            z_phys,
            half_size * 1.08,
            half_size * 1.08,
            label,
            fontsize=9
        )

    # Primo piano temporale
    add_plane(
        z_time_1,
        "T1",
        facecolor="tab:orange",
        alpha=0.25
    )

    # Piani di tracciamento
    for i, z in enumerate(z_planes):
        add_plane(
            z,
            f"Tracker {i}",
            facecolor="tab:blue",
            alpha=0.18
        )

    # Secondo piano temporale
    add_plane(
        z_time_2,
        "T2",
        facecolor="tab:orange",
        alpha=0.25
    )

    # Disegna alcune tracce
    if x_hits is not None and y_hits is not None:
        n_tracks = x_hits.shape[0]
        n_show = min(n_tracks_show, n_tracks)

        for i in range(n_show):
            ax.plot(
                z_planes,      # asse X grafico = z fisico
                x_hits[i],     # asse Y grafico = x fisico
                y_hits[i],     # asse Z grafico = y fisico
                marker="o",
                markersize=3,
                linewidth=1,
                alpha=0.8
            )

    # Limiti degli assi
    z_min_plot = z_time_1 - 0.1 * abs(z_time_2 - z_time_1)
    z_max_plot = z_time_2 + 0.1 * abs(z_time_2 - z_time_1)

    ax.set_xlim(z_min_plot, z_max_plot)
    ax.set_ylim(-half_size, half_size)
    ax.set_zlim(-half_size, half_size)

    ax.set_xlabel("z fisico [mm]")
    ax.set_ylabel("x fisico [mm]")
    ax.set_zlabel("y fisico [mm]")

    ax.set_title("Schema 3D del detector con asse z orizzontale")

    # Vista scelta per mostrare z orizzontale
    ax.view_init(elev=21, azim=-70)
    ax.text2D(0.02, 0.95, label, transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.show()

    return fig

class reward_calculator:
    def __init__(self, mode='length'): # Fixed typo 'lenght' -> 'length'
        self.mode = mode

    def forward(self, detector_state, beta_dist, theta_dist, spatial_res=None, tof_margin=None, temporal_res=None):
        if self.mode == 'length':
            return self.length_reward_incentive(detector_state, beta_dist, theta_dist)
        elif self.mode == 'length_plane':
            return self.length_plane_reward(detector_state, beta_dist, theta_dist)
        elif self.mode == 'length_params':
            return self.length_and_params_reward(detector_state, beta_dist, theta_dist, spatial_res, tof_margin, temporal_res)
            
    def length_reward_incentive(self, detector_state, beta_dist, theta_dist):
        if len(detector_state) < 2:
            cost = 0.0
        else:
            cost = detector_state[-1] * 0.01
        reward = 50.0 - beta_dist / 100.0 - theta_dist / 100.0 - cost
        reward += 10.0 if beta_dist < 20 else (-10.0 if beta_dist > 40 else 0.0)
        reward += 10.0 if theta_dist < 20 else (-10.0 if theta_dist > 40 else 0.0)
        return reward, cost

    def length_plane_reward(self, detector_state, beta_dist, theta_dist):
        if len(detector_state) < 2:
            cost = 0.0
        else:
            cost = detector_state[-1] * 0.01 + len(detector_state) * 0.2  # Fixed: self.len -> len
        reward = 50.0 - beta_dist / 100.0 - theta_dist / 100.0 - cost
        reward += 10.0 if beta_dist < 20 else (-10.0 if beta_dist > 40 else 0.0)
        reward += 10.0 if theta_dist < 20 else (-10.0 if theta_dist > 40 else 0.0)
        return reward, cost

    def length_and_params_reward(self, detector_state, beta_dist, theta_dist, spatial_res, tof_margin, time_res):
        if len(detector_state) < 2:
            cost = 0.0
        else:
            cost = detector_state[-1] * 0.01 + len(detector_state) * 0.2 + spatial_res/0.01 + tof_margin * 0.01 + time_res / 0.1  
        reward = 50.0 - beta_dist / 100.0 - theta_dist / 100.0 - cost
        reward += 10.0 if beta_dist < 20 else (-10.0 if beta_dist > 40 else 0.0)
        reward += 10.0 if theta_dist < 20 else (-10.0 if theta_dist > 40 else 0.0)
        return reward, cost


from torch.utils.tensorboard import SummaryWriter
from skrl.trainers.torch import StepTrainer
import numpy as np

#Write description on tensorboard data, Qvalue, timesteps and so on, comment and explanations
#Write different versions of the whole system, such as Reward on Termination, and potential reward shaping, and so on
class customTrainer(StepTrainer):
    def __init__(self, env, agent, cfg, logdir='runs', health_logs=False, health_dir='health-logs/', health_filename='unnamed-logs', log_interval=10):
        super().__init__(env=env, agents=agent, cfg=cfg)
        self.writer = SummaryWriter(log_dir="./runs")
        self.env = env
        self.logger = None
        self.agent = agent
        self.cfg = cfg

        logging.basicConfig(
            filename=health_dir+cfg.experiment.experiment_name,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            filemode="w"
        )
        if health_logs:
            self.logger = logging.getLogger("NetworkHealth")
            self.log_interval = log_interval
            self.health_logs = health_logs
        else:
            self._real_log_health = lambda *args, **kwargs : None

    def train(self):
        self.best_reward = -np.inf

        for timestep in range(self.cfg.timesteps):
            super().train()
            
            obs = self.env._obs_and_metrics()
            reward, cost = self.env.calculate_reward(self.env.detector_state, obs[1], obs[2], self.env.spatial_res, self.env.tof_margin, self.env.time_res)
            
            #print best detector state and reward if reward is better than previous best
            if reward > self.best_reward:
                log_msg = (
                        f"[BEST REWARD] Timestep: {timestep} | "
                        f"Best Reward: {self.best_reward:.4f} | "
                        f"Detector State: {self.env.detector_state}"
                    )
                self.logger.info(log_msg)
                self.best_reward = reward
                if(self.env.detector_state is not None and len(self.env.detector_state) > 1):
                    self.writer.add_figure("Best Detector State", draw_detector(n_planes=len(self.env.detector_state), z_planes=self.env.detector_state, plane_size=100.0, tof_margin=self.env.tof_margin, x_hits=obs[3], y_hits=obs[4], n_tracks_show=self.env.n_tracks, 
                                            label=f"Detector 3D - Reward: {reward:.2f}, Planes: {len(self.env.detector_state)}\ndist theta: {obs[1]:.2f}, dist beta: {obs[2]:.2f}\ncost: {cost:.2f}, \nSpatial_res {self.env.spatial_res:.2f}, \nTof_margin {self.env.tof_margin:.2f}, \nTime_res {self.env.time_res:.2f}"), global_step=timestep)

            if timestep % self.log_interval == 0:
                self._real_log_health(timestep=timestep, reward=reward, cost=cost)


    def _real_log_health(self, timestep, reward, cost):
        health_data = self._get_network_health()
        
        health_log = (
            f"Timestep: {timestep} | "
            f"Reward: {reward:.4f} | "
            f"Cost: {cost:.4f} | "
            f"Actor Grad Norm: {health_data['actor_grad_norm']:.4f} | "
            f"Critic Grad Norm: {health_data['critic_grad_norm']:.4f} | "
            f"Actor Loss: {health_data['actor_loss']} | "
            f"Critic Loss: {health_data['critic_loss']}"
        )

        self.logger.info(health_log)
        
        self.writer.add_scalar("Performance/Reward", reward, timestep)
        self.writer.add_scalar("Performance/Cost", cost, timestep)
        self.writer.add_scalar("Health/ActorGradNorm", health_data['actor_grad_norm'], timestep)
        self.writer.add_scalar("Health/CriticGradNorm", health_data['critic_grad_norm'], timestep)

    def _get_network_health(self):
        actor_grad_norm = 0.0
        critic_grad_norm = 0.0
        
        try:
            if hasattr(self.agent, "policy") and self.agent.policy is not None:
                actor_grads = [p.grad.norm().item() for p in self.agent.policy.parameters() if p.grad is not None]
                if actor_grads:
                    actor_grad_norm = float(np.mean(actor_grads))
            
            if hasattr(self.agent, "value") and self.agent.value is not None:
                critic_grads = [p.grad.norm().item() for p in self.agent.value.parameters() if p.grad is not None]
                if critic_grads:
                    critic_grad_norm = float(np.mean(critic_grads))
        except Exception as e:
            self.logger.error(f"Errore durante il calcolo delle norme dei gradienti: {e}")

        tracking_dict = getattr(self.agent, "tracking", {})
        actor_loss = tracking_dict.get("loss / actor loss", 0.0)
        critic_loss = tracking_dict.get("loss / critic loss", 0.0)

        return {
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss
        }