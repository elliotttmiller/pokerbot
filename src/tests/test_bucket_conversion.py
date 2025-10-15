import unittest
from deepstack.nn.bucket_conversion import convert_to_bucket

class TestBucketConversion(unittest.TestCase):
    def test_convert_to_bucket(self):
        self.assertIsNone(convert_to_bucket('AhKd'))

if __name__ == "__main__":
    unittest.main()
