import os
from tqdm import tqdm
import csv

from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import DATASET_PATHS, VAL_SIZE


class UTKFace(PairwiseComparisonDataset):

    def __init__(
            self, 
            attribute='age', 
            transform=None, 
            supports_scores=True,
            split='train',
            dataset_name='utkface',
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
        
        train_split_file = DATASET_PATHS[self.dataset_name]['train_split_file']
        test_split_file = DATASET_PATHS[self.dataset_name]['test_split_file']

        with open(train_split_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]
            indices = [i for i in range(len(rows))]
            self.rng.shuffle(indices)
            num_val = int(VAL_SIZE * len(rows))
            val_paths = [os.path.join('images', rows[i][1].replace('.chip.jpg', '')) for i in indices[:num_val]]
            train_paths = [os.path.join('images', rows[i][1].replace('.chip.jpg', '')) for i in indices[num_val:]]

        with open(test_split_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]
            test_paths = [os.path.join('images', rows[i][1].replace('.chip.jpg', '')) for i in range(len(rows))]

        return train_paths, val_paths, test_paths

    def _create_labels(self):
        
        scores = []
        for img_path in tqdm(self.img_paths):
            score = int(img_path.split('/')[-1].split('_')[0])
            scores.append(score)

        if self.label_mode == 'regression':
            self.labels = scores

        elif self.label_mode == 'classification':
            self.labels = self._discretize_scores(scores, num_bins=10)


if __name__ == '__main__':
    from rankbench.constants import DATASET_PATHS
    dataset = UTKFace(
        attribute='age',
        split='train',
        label_mode='classification',
        use_pairs=False,
        use_val_set_for_training=True,
    )
    print(len(dataset))
    dataset.visualize_examples(n=8)
    # dataset.visualize_pairs(n=5)