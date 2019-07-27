from data_loader_from_git import DataLoader
import os

class TestDataLoader:

    def test_data_loader(self):
        dataloader = DataLoader('https://github.com/taivop/joke-dataset.git', 'datasets/jokes')
        assert os.path.exists('datasets/jokes')