import unittest
from deepstack.nn.cpu_gpu_model_converter import convert_model

class TestCPUGPUModelConverter(unittest.TestCase):
    def test_convert_model(self):
        self.assertIsNone(convert_model('model'))

if __name__ == "__main__":
    unittest.main()
