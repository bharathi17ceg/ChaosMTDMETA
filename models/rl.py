# models/reinforcement_learning.py
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from gym import spaces

class MTDEnv(gym.Env):
    def __init__(self, network_state_size=8):
        super(MTDEnv, self).__init__()
        self.action_space = spaces.Discrete(8)  # 8 MTD strategies
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(network_state_size,), dtype=np.float32)
        
        self.current_state = None
        self.mtd_engine = LorenzMTD()
        self.network = SDNNetwork()

    def reset(self):
        self.current_state = self.network.get_state()
        return self.current_state

    def step(self, action):
        # Execute MTD action
        reward = self._apply_mtd_action(action)
        next_state = self.network.get_state()
        done = False
        
        return next_state, reward, done, {}

    def _apply_mtd_action(self, action):
        """Map action to MTD strategy and calculate reward"""
        strategy_map = {
            0: self._path_mutation,
            1: self._port_mutation,
            2: self._ip_mutation,
            3: self._path_port_mutation,
            4: self._path_ip_mutation,
            5: self._ip_port_mutation,
            6: self._full_mutation,
            7: self._no_action
        }
        
        cost = strategy_map[action]()
        effectiveness = self.network.security_improvement()
        return effectiveness - cost

    def _path_mutation(self):
        self.network.apply_path_mutation(
            self.mtd_engine.mutate_path(self.network.topology))
        return 0.2

    def _port_mutation(self):
        self.network.apply_port_mutation(
            self.mtd_engine.mutate_port(self.network.ports))
        return 0.3

    def _ip_mutation(self):
        self.network.apply_ip_mutation(
            self.mtd_engine.mutate_ip(self.network.ip_addresses))
        return 0.4

    def _path_port_mutation(self):
        self.network.apply_path_mutation(
            self.mtd_engine.mutate_path(self.network.topology))
        self.network.apply_port_mutation(
            self.mtd_engine.mutate_port(self.network.ports))
        return 0.5

    def _path_ip_mutation(self):
        self.network.apply_path_mutation(
            self.mtd_engine.mutate_path(self.network.topology))
        self.network.apply_ip_mutation(
            self.mtd_engine.mutate_ip(self.network.ip_addresses))
        return 0.6

    def _ip_port_mutation(self):
        self.network.apply_ip_mutation(
            self.mtd_engine.mutate_ip(self.network.ip_addresses))
        self.network.apply_port_mutation(
            self.mtd_engine.mutate_port(self.network.ports))
        return 0.7

    def _full_mutation(self):
        self.network.apply_path_mutation(
            self.mtd_engine.mutate_path(self.network.topology))
        self.network.apply_port_mutation(
            self.mtd_engine.mutate_port(self.network.ports))
        self.network.apply_ip_mutation(
            self.mtd_engine.mutate_ip(self.network.ip_addresses))
        return 0.8

    def _no_action(self):
        return 0.0


class RLAgent:
    def __init__(self, env):
        self.model = A2C('MlpPolicy', env, verbose=1,
                        learning_rate=0.0003,
                        n_steps=128,
                        gamma=0.99)

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = A2C.load(path)
