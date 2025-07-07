import csv
import os
from tqdm import tqdm
import numpy as np

from rankbench.constants import DATASET_SPLIT_SEED, VAL_SIZE
from rankbench.datasets.base import PairwiseComparisonDataset


class BoneAge(PairwiseComparisonDataset):

    def __init__(
        self, 
        images_dir, 
        attribute='boneage', 
        transform=None, 
        split='train',
    ):
        assert attribute == 'boneage', f"Invalid attribute: {attribute}"
        super().__init__(
            images_dir=images_dir,
            attribute=attribute,
            transform=transform,
            split=split,
            dataset_name='bone_age',
            supports_scores=True
        )
    
    def _create_img_paths(self):
        self.img_paths = []
        dir_to_use = 'boneage-training-dataset' if self.split in ['train', 'val'] else 'boneage-validation-dataset'
        dir_to_use = os.path.join(self.images_dir, dir_to_use)
        for f in os.listdir(dir_to_use):
            if f.lower().endswith(".png"):
                self.img_paths.append(os.path.join(dir_to_use, f))
        print(f"Found {len(self.img_paths)} images in {dir_to_use}")

        labels_file = 'train.csv' if self.split in ['train', 'val'] else 'Validation Dataset.csv'

        with open(os.path.join(self.images_dir, 'labels', labels_file), 'r') as f:
            reader = csv.reader(f)
            labels = list(reader)[1:]
            idx_to_label = {}
            for item in labels:
                idx_to_label[int(item[0])] = float(item[2]) if self.split == 'test' else float(item[1])

        if self.split in ['train', 'val']:

            rng = np.random.default_rng(DATASET_SPLIT_SEED)
            rng.shuffle(self.img_paths)

            num_val = int(VAL_SIZE * len(self.img_paths))

            if self.split == 'val':
                self.img_paths = self.img_paths[:num_val]
            elif self.split == 'train':
                self.img_paths = self.img_paths[num_val:]
        
        self.labels = []
        for img_path in self.img_paths:
            idx = int(img_path.split('/')[-1].split('.')[0])
            self.labels.append(idx_to_label[idx])

    
    def _make_pairwise_comparisons(self):

        self.pairs = []
        self.ties = []
        
        N = len(self.img_paths)
        for i in tqdm(range(N)):
            label_i = self.labels[i]
            for j in range(i+1, N):
                label_j = self.labels[j]
                if label_i < label_j:
                    label = 0
                elif label_i > label_j:
                    label = 1
                else:
                    self.ties.append((i, j))
                    continue
                self.pairs.append((i, j, label))
        
        print(f"Generated {len(self.pairs)} pairwise comparisons")
        print(f"Detected {len(self.ties)} ties")


if __name__ == '__main__':
    from rankbench.constants import DATASET_PATHS
    dataset = BoneAge(images_dir=DATASET_PATHS['bone_age']['images_dir'])
    dataset.visualize_examples()
    dataset.visualize_pairs()