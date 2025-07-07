import os
from scipy.io import loadmat    
import numpy as np
import matplotlib.pyplot as plt
from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import VAL_SIZE


class UCF_QNRF(PairwiseComparisonDataset):

    def __init__(
        self, 
        attribute='crowd_count', 
        transform=None,
        supports_scores=True,
        split='train',
        dataset_name='ucf_qnrf',
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
    
    def _create_img_paths(self):
        
        trainval_dir = os.path.join(self.data_dir, 'Train')
        test_dir = os.path.join(self.data_dir, 'Test')
        trainval_paths = [os.path.join('Train', f) for f in os.listdir(trainval_dir) if f.endswith('.jpg')]
        test_paths = [os.path.join('Test', f) for f in os.listdir(test_dir) if f.endswith('.jpg')]

        self.rng.shuffle(trainval_paths)
        num_val = int(VAL_SIZE * len(trainval_paths))
        train_paths = trainval_paths[num_val:]
        val_paths = trainval_paths[:num_val]

        return train_paths, val_paths, test_paths

    def _create_labels(self):

        scores = []
        label_paths = [os.path.join(self.data_dir, f.replace('.jpg', '_ann.mat')) for f in self.img_paths]
        for img_path, label_path in zip(self.img_paths, label_paths):
            assert os.path.exists(img_path), f"Image path does not exist: {img_path}"
            assert os.path.exists(label_path), f"Label path does not exist: {label_path}"
            count = loadmat(label_path)['annPoints'].shape[0]
            scores.append(count)

        if self.label_mode == 'regression':
            self.labels = scores

        elif self.label_mode == 'classification':
            self.labels = self._discretize_scores(scores, num_bins=10)


if __name__ == '__main__':
    test_dataset = UCF_QNRF(split='train', label_mode='regression')

    test_dataset._make_histogram('train_histogram.png')
    test_dataset.visualize_examples(n=8)