import os
from tqdm import tqdm
import csv
from PIL import Image
import pandas as pd
from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import DATASET_PATHS, VAL_SIZE


class Koniq10k(PairwiseComparisonDataset):

    def __init__(
            self, 
            attribute='good_looking_image', 
            transform=None, 
            supports_scores=True,
            split='train',
            dataset_name='koniq_10k',
            use_pairs=False,
            label_mode='regression',
            logger=None,
            use_val_set_for_training=False,
        ):
        self.images_dir = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], '512x384')
        self.labels_file = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'koniq10k_distributions_sets.csv')
        self.labels_df = pd.read_csv(self.labels_file)

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

        
        train_img_paths = self.labels_df[self.labels_df['set'] == 'training']['image_name'].tolist()
        val_img_paths = self.labels_df[self.labels_df['set'] == 'validation']['image_name'].tolist()
        test_img_paths = self.labels_df[self.labels_df['set'] == 'test']['image_name'].tolist()
        
        train_img_paths = [os.path.join('512x384', img_path) for img_path in train_img_paths]
        val_img_paths = [os.path.join('512x384', img_path) for img_path in val_img_paths]
        test_img_paths = [os.path.join('512x384', img_path) for img_path in test_img_paths]

        return train_img_paths, val_img_paths, test_img_paths

    def _create_labels(self):

        labels = []
        for img_path in tqdm(self.img_paths):
            img_name = img_path.split('/')[-1]
            label_row = self.labels_df[self.labels_df['image_name'] == img_name]['MOS'].values
            assert len(label_row) == 1, f"Expected 1 label for img {img_path}, got {len(label_row)}"
            labels.append(label_row[0])
        
        if self.label_mode == 'regression':
            self.labels = labels
        elif self.label_mode == 'classification':
            self.labels = self._discretize_scores(labels, num_bins=5)
    

if __name__ == '__main__':
    dataset = Koniq10k(
        attribute='good_looking_image',
        split='test',
        use_pairs=False,
    )
    dataset.visualize_examples(n=8)