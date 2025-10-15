"""
Protocol-to-node conversion utilities for ACPC protocol.
Ported from DeepStack Lua protocol_to_node.lua.
"""
import re

class ACPCProtocolToNode:
    def __init__(self):
        pass

    def parse_actions(self, actions):
        out = []
        actions_remainder = actions
        while actions_remainder:
            if actions_remainder.startswith('c'):
                out.append({'action': 'ccall'})
                actions_remainder = actions_remainder[1:]
            elif actions_remainder.startswith('r'):
                m = re.match(r'^r(\d+)', actions_remainder)
                raise_amount = int(m.group(1)) if m else None
                out.append({'action': 'raise', 'raise_amount': raise_amount})
                actions_remainder = actions_remainder[len(m.group(0)):] if m else actions_remainder[1:]
            elif actions_remainder.startswith('f'):
                out.append({'action': 'fold'})
                actions_remainder = actions_remainder[1:]
            else:
                raise ValueError('Unknown action chunk')
        return out

    def parse_state(self, state):
        # Example: MATCHSTATE:0:99:cc/r8146c/cc/cc:4cTs|Qs9s/9h5d8d/6c/6d
        # This is a stub; real parsing would extract all fields
        m = re.match(r'MATCHSTATE:(\d+):(\d+):([^:]*):([^|]*)\|(.*)', state)
        if not m:
            raise ValueError('Invalid ACPC state string')
        position = int(m.group(1))
        hand_id = int(m.group(2))
        actions_raw = m.group(3)
        board = m.group(4)
        # For simplicity, not parsing hands here
        return {
            'position': position,
            'hand_id': hand_id,
            'actions_raw': actions_raw,
            'board': board,
            'acting_player': position,  # stub
            'bet1': 0, 'bet2': 0, 'stack': 1000  # stub
        }

    def parsed_state_to_node(self, parsed_state):
        # Convert parsed state to tree node (stub)
        return parsed_state

    def action_to_message(self, last_msg, adviced_action):
        # Convert action dict to ACPC protocol message (stub)
        action = adviced_action.get('action')
        if action == 'raise':
            return f"r{adviced_action.get('raise_amount', 0)}"
        elif action == 'fold':
            return "f"
        elif action == 'ccall':
            return "c"
        else:
            return ""
    

def protocol_to_node(state, actions):
    """Convert protocol state and actions to a node object."""
    proto = ACPCProtocolToNode()
    proto.parse_state(state)
    proto.parse_actions(actions)
    return proto
