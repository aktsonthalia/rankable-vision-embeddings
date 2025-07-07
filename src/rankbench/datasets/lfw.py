import numpy as np
import os

from rankbench.datasets.base import PairwiseComparisonDataset


class LFW10(PairwiseComparisonDataset):

    def __init__(
            self, 
            attribute='smile', 
            transform=None, 
            supports_scores=False,
            split='train',
            dataset_name='lfw',
            use_pairs=True,
            label_mode='regression',
            logger=None,
            use_val_set_for_training=False,
        ):
        super().__init__(
            attribute=attribute,
            transform=transform,
            supports_scores=supports_scores,
            split=split,
            dataset_name=dataset_name,
            use_pairs=use_pairs,
            label_mode=label_mode,
            logger=logger,
            use_val_set_for_training=use_val_set_for_training,
        )
    
    def _make_img_paths_filename(self, split):
        return os.path.join(self.data_dir, 'cached_outputs', f"{self.attribute}_{split}_img_list.txt")
    
    def _create_img_paths(self):
        pass

    def _load_img_paths(self):
        with open(self._make_img_paths_filename(self.split), 'r') as f:
            img_paths = f.readlines()
            img_paths = [x.strip() for x in img_paths]
        self.img_paths = [os.path.join(self.data_dir, 'images', img_path) for img_path in img_paths]

    def _make_pairwise_comparisons(self):
        
        pairs_file = os.path.join(
            self.data_dir, 
            'cached_outputs', 
            f"{self.attribute}_{self.split}.txt"
        )
        self.pairs = []

        with open(pairs_file, 'r') as f:
            for line in f:
                idx1, idx2, label = line.strip().split()
                idx1 = int(idx1)
                idx2 = int(idx2)
                label = int(label)
                self.pairs.append((idx1, idx2, label))
        
        print(f"Generated {len(self.pairs)} pairwise comparisons")
    

if __name__ == '__main__':
    dataset = LFW10(
        attribute='goodlooking',
        dataset_name='lfw',
        split='test',
        use_pairs=True,
    )
    dataset.visualize_pairs(n=8)