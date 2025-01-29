# models/chaos_mtd.py
import numpy as np

class LorenzMTD:
    def __init__(self, sigma=10.0, rho=28.0, beta=8/3, dt=0.01):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.x = np.random.rand()
        self.y = np.random.rand()
        self.z = np.random.rand()

    def _lorenz_step(self):
        dx = self.sigma * (self.y - self.x) * self.dt
        dy = (self.x * (self.rho - self.z) - self.y) * self.dt
        dz = (self.x * self.y - self.beta * self.z) * self.dt
        
        self.x += dx
        self.y += dy
        self.z += dz
        return self.x, self.y, self.z

    def mutate_ip(self, original_ip, ip_ranges):
        """Generate mutated IP using Lorenz system"""
        octets = original_ip.split('.')
        new_octets = []
        for i, octet in enumerate(octets):
            x, _, _ = self._lorenz_step()
            new_val = int(ip_ranges[i][0] + (x % 1) * (ip_ranges[i][1] - ip_ranges[i][0]))
            new_octets.append(str(new_val))
        return ".".join(new_octets)

    def mutate_port(self, original_port, port_range=(1024, 65535)):
        """Generate mutated port using chaotic values"""
        _, y, _ = self._lorenz_step()
        return int(port_range[0] + (y % 1) * (port_range[1] - port_range[0]))

    def mutate_path(self, network_graph):
        """Generate new network path using chaotic weights"""
        _, _, z = self._lorenz_step()
        mutated_graph = []
        for edge in network_graph:
            src, dst = edge
            weight = z % 100  # Ensure weight stays within 0-100
            mutated_graph.append((src, dst, weight))
        return mutated_graph