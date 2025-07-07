import os
from tqdm import tqdm
import numpy as np

from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import DATASET_SPLIT_SEED, VAL_SIZE

class DigitsDataset(PairwiseComparisonDataset):

    def __init__(
        self, 
        images_dir, 
        attribute='high_number', 
        transform=None,
        dataset_name='digits',
        split='train'
    ):
        assert attribute == 'high_number', f"Invalid attribute: {attribute}"
        self.train_dir = os.path.join(images_dir, 'train')
        assert os.path.exists(self.train_dir), f"Invalid path: {self.train_dir}"
        self.test_dir = os.path.join(images_dir, 'test')
        assert os.path.exists(self.test_dir), f"Invalid path: {self.test_dir}"
        super().__init__(
            dataset_name=dataset_name,  
            images_dir=images_dir,
            transform=transform, 
            attribute=attribute, 
            supports_scores=True, 
            split=split
        )


    def _create_img_paths(self):
        train_dir = os.path.join(self.images_dir, 'train')
        test_dir = os.path.join(self.images_dir, 'test')

        trainval_img_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
        self.rng.shuffle(trainval_img_paths)
        num_val = int(VAL_SIZE * len(trainval_img_paths))
        train_img_paths = trainval_img_paths[:num_val]
        val_img_paths = trainval_img_paths[num_val:]

        test_img_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

        return train_img_paths, val_img_paths, test_img_paths
    
    def _create_labels(self):
        self.labels = []
        for img_path in self.img_paths:
            assert img_path.endswith('.png')
            assert os.path.exists(img_path)
            label = int(img_path.split('/')[-1].split('.')[0].split('_')[-1][0])
            self.labels.append(label)

    def _make_pairwise_comparisons(self):

        self.pairs = []
        self.ties = []

        N = len(self.img_paths)
        for i in tqdm(range(N)):
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
        print(f"Total: {len(self.pairs)} comparisons")


if __name__ == '__main__':
    from rankbench.constants import DATASET_PATHS
    dataset = DigitsDataset(images_dir=DATASET_PATHS['digits']['images_dir'], split='val')
    dataset.visualize_examples()
    dataset.visualize_pairs()