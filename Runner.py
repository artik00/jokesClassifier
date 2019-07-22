import pickle
import os
from DataLoader import DataLoader

if __name__ == "__main__":
    dataset_dir = 'dataset/jokes'
    data_loader = DataLoader('https://github.com/CrowdTruth/Short-Text-Corpus-For-Humor-Detection.git', dataset_dir)
    filename = 'humorous_oneliners.pickle'
    oneliners_pickle = os.path.join(dataset_dir, 'datasets', filename)
    with open(oneliners_pickle, 'rb') as f:
        oneliners = pickle.load(f, encoding='bytes')
        print(len(oneliners))
        [print(x) for x in oneliners]
