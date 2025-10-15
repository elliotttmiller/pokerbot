"""
ACPC game protocol utilities for DeepStack poker AI.
Ported from DeepStack Lua acpc_game.lua.
"""
from deepstack.protocol.network_communication import ACPCNetworkCommunication
from deepstack.protocol.protocol_to_node import ACPCProtocolToNode

class ACPCGame:
    def __init__(self):
        self.protocol_to_node = ACPCProtocolToNode()
        self.network_communication = None
        self.last_msg = None

    def connect(self, server, port):
        self.network_communication = ACPCNetworkCommunication()
        self.network_communication.connect(server, port)

    def get_next_situation(self):
        while True:
            msg = self.network_communication.get_line()
            print("Received ACPC dealer message:", msg)
            parsed_state = self.protocol_to_node.parse_state(msg)
            if parsed_state['acting_player'] == parsed_state['position']:
                if parsed_state['bet1'] == parsed_state['bet2'] and parsed_state['bet1'] == parsed_state['stack']:
                    print("Not our turn - allin")
                else:
                    print("Our turn")
                    self.last_msg = msg
                    node = self.protocol_to_node.parsed_state_to_node(parsed_state)
                    return parsed_state, node
            else:
                print("Not our turn")

    def play_action(self, adviced_action):
        message = self.protocol_to_node.action_to_message(self.last_msg, adviced_action)
        print("Sending a message to the ACPC dealer:", message)
        self.network_communication.send_line(message)
