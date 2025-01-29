# File 1: mtd_module.py
import numpy as np

class LorenzMTD:
    def __init__(self, ryapp):
        super().__init__()
        self.network = SDNNetwork(ryapp)
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32)
        
    def step(self, action):
        # Execute action on real SDN network
        reward = self._apply_mtd_action(action)
        next_state = self.network.get_state()
        done = False
        return next_state, reward, done, {}
    def __init__(self):
        self.sigma, self.rho, self.beta, self.lambd = 10, 28, 8/3, 2.0
        self.x, self.y, self.z, self.w = 0.1, 0.0, 0.0, 0.0

    def chaotic_system(self, dt=0.01):
        dx = self.sigma * (self.y - self.x) * dt
        dy = (self.x * (self.rho - self.z) - self.y) * dt
        dz = (self.x * self.y - self.beta * self.z) * dt
        dw = self.lambd * (self.z - self.w) * dt
        self.x += dx
        self.y += dy
        self.z += dz
        self.w += dw
        return self.x, self.y, self.z, self.w

    def mutate_ip(self, real_ip, ip_ranges):
        x, y, z, w = self.chaotic_system()
        octets = [
            int(np.clip(abs(int(x * 255) % 256), ip_ranges[0][0], ip_ranges[0][1])),
            int(np.clip(abs(int(y * 255) % 256), ip_ranges[1][0], ip_ranges[1][1])),
            int(np.clip(abs(int(z * 255) % 256), ip_ranges[2][0], ip_ranges[2][1])),
            int(np.clip(abs(int(w * 255) % 256), ip_ranges[3][0], ip_ranges[3][1]))
        ]
        return ".".join(map(str, octets))

    def mutate_port(self, port_range):
        x, y, _, _ = self.chaotic_system()
        return (
            int(np.clip(abs(int(x * 1000) % port_range[1]), port_range[0], port_range[1])),
            int(np.clip(abs(int(y * 1000) % port_range[1]), port_range[0], port_range[1]))
        )

    def mutate_path(self, network_graph):
        _, y, z, _ = self.chaotic_system()
        mutated_paths = []
        for path in network_graph:
            new_path = [path[0]]
            for node in path[1:]:
                weight = np.clip(abs(int(y * 10) % 100), 1, 100)
                mutated_paths.append((new_path[-1], node, weight))
                new_path.append(node)
        return mutated_paths
