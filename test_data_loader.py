from DataLoader import DataLoader
import os

class TestDataLoader:

    def test_data_loader(self):
        dataloader = DataLoader('https://github.com/taivop/joke-dataset.git', 'datasets')
        assert os.path.exists('datasets')