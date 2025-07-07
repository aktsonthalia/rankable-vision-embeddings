import os
from scipy.io import loadmat    
import numpy as np
import matplotlib.pyplot as plt
from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import VAL_SIZE


class NWPU_Crowd(PairwiseComparisonDataset):

    def __init__(
        self, 
        attribute='count', 
        transform=None,
        supports_scores=True,
        split='train',
        dataset_name='nwpu_crowd',
        use_pairs=False,
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
    
    def _instances_to_paths(self, split):
        instances_path = os.path.join(self.data_dir, 'NWPU-Crowd', 'train.txt')
        img_paths = []
        with open(instances_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = f'{line.split(" ")[0]}.jpg'
                img_paths.append(os.path.join(self.data_dir, img_path))

        return img_paths

    def _create_img_paths(self):
        
        train_img_paths = self._instances_to_paths('train')
        val_img_paths = self._instances_to_paths('val')
        test_img_paths = self._instances_to_paths('test')

        return train_img_paths, val_img_paths, test_img_paths

    def _create_labels(self):

        scores = []
        label_paths = [os.path.join(self.data_dir, f.replace('.jpg', '.mat')) for f in self.img_paths]
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
    test_dataset = NWPU_Crowd(split='test', label_mode='regression')

    test_dataset._make_histogram('train_histogram.png')
    test_dataset.visualize_examples(n=8)