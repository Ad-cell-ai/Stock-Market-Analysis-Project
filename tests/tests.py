import unittest
import pandas as pd
from src.features import add_features

class TestFeatures(unittest.TestCase):
    def test_add_features(self):
        df = pd.DataFrame({"Close": [100,101,102,103,104], "Volume":[10,20,30,40,50]}, 
                          index=pd.date_range("2020-01-01", periods=5))
        result = add_features(df)
        self.assertIn("rsi", result.columns)

if __name__ == "__main__":
    unittest.main()
