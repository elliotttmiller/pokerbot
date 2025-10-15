import unittest
from deepstack.protocol.protocol_to_node import protocol_to_node

class TestProtocolToNode(unittest.TestCase):
    def test_protocol_to_node(self):
        self.assertIsNone(protocol_to_node('MATCHSTATE:...'))

if __name__ == "__main__":
    unittest.main()
