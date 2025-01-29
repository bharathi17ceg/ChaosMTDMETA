#  ryu_app.py 
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4
from meta_learner import MetaAnomalyDetector
from mtd_module import LorenzMTD
from rl_agent import RLAgent
import numpy as np

class IDSController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    def __init__(self):
        super().__init__()
        self.network = SDNNetwork(self)
        self.topology_api_app = self
        
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        self.network._detect_topology()
        
    @set_ev_cls(event.EventLinkAdd)
    def link_add_handler(self, ev):
        self.network._detect_topology()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = MetaAnomalyDetector()
        self.mtd_engine = LorenzMTD()
        self.rl_agent = RLAgent(MTDEnv())
        self.network_state = np.zeros(5)  # State: [bandwidth, latency, packet_loss, etc.]
        
        # Full action mapping with strategy combinations
        self.mtd_actions = {
            0: self._apply_path_mutation,
            1: self._apply_port_mutation,
            2: self._apply_ip_mutation,
            3: self._apply_path_port_mutation,
            4: self._apply_path_ip_mutation,
            5: self._apply_ip_port_mutation,
            6: self._apply_comprehensive_mutation,
            7: self._apply_no_action
        }

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        pkt = packet.Packet(msg.data)
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        
        if ip_pkt:
            # 1. Extract features and detect anomalies
            features = self._extract_features(ip_pkt, msg)
            anomalies = self.detector.detect_anomalies(features)
            
            if 'BENIGN' not in anomalies:
                self.logger.info(f"Attack detected: {', '.join(anomalies)}")
                
                # 2. Update network state
                self._update_network_state(anomalies)
                
                # 3. Get RL-based MTD action
                action = self.rl_agent.select_action(self.network_state)
                
                # 4. Apply selected MTD strategy
                self.mtd_actions[action](ip_pkt, msg.datapath)

    def _extract_features(self, ip_pkt, msg):
        """Enhanced feature extraction for RL state"""
        return np.array([
            msg.msg_len,                        # Flow duration
            ip_pkt.total_length,                # Total fwd packets
            ip_pkt.identification,              # Flow packets/s
            ip_pkt.ttl,                         # TTL
            ip_pkt.header_length,               # Header length
            msg.buffer_id,                      # Buffer usage
            len(msg.data),                      # Packet size
            ip_pkt.offset                       # Fragment offset
        ])

    def _update_network_state(self, anomalies):
        """Update network state vector for RL"""
        # Implement actual network monitoring here
        self.network_state = np.random.rand(5)  

    # MTD Action Implementations
    def _apply_path_mutation(self, pkt, datapath):
        new_path = self.mtd_engine.mutate_path(self._get_network_topology())
        self._update_flow_rules(datapath, pkt, path=new_path)
        self.logger.info("Applied path mutation")

    def _apply_port_mutation(self, pkt, datapath):
        new_src, new_dst = self.mtd_engine.mutate_port()
        self._update_flow_rules(datapath, pkt, src_port=new_src, dst_port=new_dst)
        self.logger.info("Applied port mutation")

    def _apply_ip_mutation(self, pkt, datapath):
        new_ip = self.mtd_engine.mutate_ip(pkt.src)
        self._update_flow_rules(datapath, pkt, ip=new_ip)
        self.logger.info("Applied IP mutation")

    def _apply_path_port_mutation(self, pkt, datapath):
        self._apply_path_mutation(pkt, datapath)
        self._apply_port_mutation(pkt, datapath)
        self.logger.info("Applied path+port mutation")

    def _apply_path_ip_mutation(self, pkt, datapath):
        self._apply_path_mutation(pkt, datapath)
        self._apply_ip_mutation(pkt, datapath)
        self.logger.info("Applied path+IP mutation")

    def _apply_ip_port_mutation(self, pkt, datapath):
        self._apply_ip_mutation(pkt, datapath)
        self._apply_port_mutation(pkt, datapath)
        self.logger.info("Applied IP+port mutation")

    def _apply_comprehensive_mutation(self, pkt, datapath):
        self._apply_path_mutation(pkt, datapath)
        self._apply_ip_mutation(pkt, datapath)
        self._apply_port_mutation(pkt, datapath)
        self.logger.info("Applied comprehensive mutation")

    def _apply_no_action(self, pkt, datapath):
        self.logger.info("No MTD action taken")

    def _update_flow_rules(self, datapath, pkt, **mutations):
        """Update OpenFlow rules based on mutations"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Remove old rules
        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=pkt.src,
            ipv4_dst=pkt.dst
        )
        self._delete_flow(datapath, match)
        
        # Add new rules with mutations
        actions = []
        if 'ip' in mutations:
            actions.append(parser.OFPActionSetField(ipv4_src=mutations['ip']))
        if 'port' in mutations:
            actions.append(parser.OFPActionSetField(tcp_src=mutations['src_port']))
            actions.append(parser.OFPActionSetField(tcp_dst=mutations['dst_port']))
        
        actions.append(parser.OFPActionOutput(ofproto.OFPP_NORMAL))
        
        self._install_flow(datapath, match, actions)

    def _delete_flow(self, datapath, match):
        """Delete existing flow rules"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        flow_mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            match=match,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY
        )
        datapath.send_msg(flow_mod)

    def _install_flow(self, datapath, match, actions, priority=1):
        """Install new flow rules"""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst
        )
        datapath.send_msg(mod)
        
        
if __name__ == "__main__":    
    app = IDSController()




