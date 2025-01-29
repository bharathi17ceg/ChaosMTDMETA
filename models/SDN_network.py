# models/sdn_network.py
from ryu.controller import dpset
from ryu.topology import api as topology_api
import time

class SDNNetwork:
    def __init__(self, ryapp):
        self.ryu_app = ryapp  # Reference to Ryu controller app
        self.state = {
            'bandwidth': 0,
            'latency': 0,
            'packet_loss': 0,
            'jitter': 0,
        }
        self.last_update = time.time()
        self.topology = {}

    def _get_switch_statistics(self):
        """Get statistics from all switches using Ryu API"""
        stats = []
        for dp in self.ryu_app.dpset.dps.values():
            stats.append({
                'port_stats': self.ryu_app.get_port_stats(dp),
                'flow_stats': self.ryu_app.get_flow_stats(dp),
                'aggregate_stats': self.ryu_app.get_aggregate_flow_stats(dp)
            })
        return stats

    def _calculate_bandwidth(self, port_stats):
        """Calculate bandwidth utilization from port statistics"""
        total_bytes = sum(p.rx_bytes + p.tx_bytes for p in port_stats)
        time_diff = time.time() - self.last_update
        return total_bytes / (time_diff * 1e6)  # Convert to Mbps

    def _detect_topology(self):
        """Build network topology using Ryu's topology service"""
        self.topology = {
            'switches': topology_api.get_all_switches(self.ryu_app),
            'links': topology_api.get_all_link(self.ryu_app),
            'hosts': topology_api.get_all_host(self.ryu_app)
        }

    def get_state(self):
        """Get current network state from SDN controller"""
        # 1. Get switch statistics
        stats = self._get_switch_statistics()
        
        # 2. Calculate bandwidth utilization
        port_stats = [p for s in stats for p in s['port_stats']]
        self.state['bandwidth'] = self._calculate_bandwidth(port_stats)
        
        # 3. Get flow counts
        self.state['jitter'] = sum(len(s['jitter']) for s in stats)
        
        # 4. Update topology
        self._detect_topology()
        
       
        
        return np.array([
            self.state['bandwidth'],
            self.state['latency'],
            self.state['packet_loss'],
            self.state['jitter'],
            
        ])

    def apply_path_mutation(self, new_path):
        """Implement path mutation using SDN flow rules"""
        ofp_parser = self.ryu_app.ofp_parser
        ofp = self.ryu_app.ofproto
        
        # 1. Remove old flows
        for switch in self.topology['switches']:
            match = ofp_parser.OFPMatch()
            self.ryu_app.delete_flow(switch.dp, match)
        
        # 2. Install new path rules
        for hop in new_path:
            match = ofp_parser.OFPMatch(in_port=hop['in_port'])
            actions = [ofp_parser.OFPActionOutput(hop['out_port'])]
            self.ryu_app.install_flow(
                hop['switch'].dp, 
                match=match,
                actions=actions,
                priority=1000
            )
