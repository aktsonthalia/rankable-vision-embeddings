import os
import matplotlib.pyplot as plt
from scipy.io import loadmat    
import numpy as np
from tqdm import tqdm

from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import DATASET_PATHS, DATASET_SPLIT_SEED


class UCFCC50Dataset(PairwiseComparisonDataset):

    def __init__(
        self, 
        images_dir, 
        attribute='count', 
        transform=None,
        dataset_name='ucf_cc_50',
        split='train'
    ):
        super().__init__(
            dataset_name=dataset_name,
            images_dir=images_dir,
            transform=transform,
            attribute=attribute,
            supports_scores=True,
            split=split
        )
    
    def _create_img_paths(self):
        n = len([f for f in os.listdir(self.images_dir) if f.endswith('.jpg')])
        assert n == 50
        self.img_filenames = [f"{i}.jpg" for i in range(1, n + 1)]

        rng = np.random.default_rng(DATASET_SPLIT_SEED)
        rng.shuffle(self.img_filenames)
        num_train = DATASET_PATHS['ucf_cc_50']['num_train']
        num_val = DATASET_PATHS['ucf_cc_50']['num_val']

        if self.split == 'train':
            self.img_filenames = self.img_filenames[:num_train]
        elif self.split == 'val':
            self.img_filenames = self.img_filenames[num_train:num_train + num_val]
        elif self.split == 'test':
            self.img_filenames = self.img_filenames[num_train + num_val:]

        self.img_paths = [os.path.join(self.images_dir, f) for f in self.img_filenames]
        self.labels = []
        label_paths = [os.path.join(self.images_dir, f.replace('.jpg', '_ann.mat')) for f in self.img_filenames]
        for img_path, label_path in zip(self.img_paths, label_paths):
            assert os.path.exists(img_path)
            assert os.path.exists(label_path)
            count = loadmat(label_path)['annPoints'].shape[0]
            self.labels.append(count)
        
    def _make_pairwise_comparisons(self):

        self.pairs = []
        self.ties = []
        N = len(self.img_filenames)
        for i in range(N):
            for j in range(i+1, N):
                if self.labels[i] < self.labels[j]:
                    label = 0
                elif self.labels[i] > self.labels[j]:
                    label = 1
                else:
                    self.ties.append((i, j))
                    continue
                self.pairs.append((i, j, label))
        
        print(f"Generated {len(self.pairs)} pairwise comparisons")
        print(f"Detected {len(self.ties)} ties")
        print(f"Total: {len(self.pairs) + len(self.ties)} comparisons")

        total_pairs = N * (N - 1) // 2
        assert len(self.pairs) + len(self.ties) == total_pairs

if __name__ == '__main__':
    dataset = UCFCC50Dataset(images_dir=DATASET_PATHS['ucf_cc_50']['images_dir'], split='val')
    dataset.visualize_examples()
    dataset.visualize_pairs()