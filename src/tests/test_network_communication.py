import unittest
from deepstack.protocol.network_communication import send_message

class TestNetworkCommunication(unittest.TestCase):
    def test_send_message(self):
        self.assertIsNone(send_message('MATCHSTATE:...'))

if __name__ == "__main__":
    unittest.main()
