import os
from tqdm import tqdm
import csv
import json
from PIL import Image

from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import DATASET_PATHS, VAL_SIZE


class AVA(PairwiseComparisonDataset):

    def __init__(
            self, 
            attribute='good_looking_image', 
            transform=None, 
            supports_scores=True,
            split='train',
            dataset_name='ava',
            use_pairs=False,
            label_mode='regression',
            logger=None,
            use_val_set_for_training=False,
        ):
        self.images_dir = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'images')
        self.image_lists_dir = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'aesthetics_image_lists')
        self.labels_file = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'labels.txt')
        self._create_idx_to_mos()

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

    def _create_idx_to_mos(self):
        
        with open(self.labels_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        self.idx_to_mos = {}
        for row in tqdm(rows):
            row = row[0].split(' ')
            row = [int(x) for x in row]
            idx = row[1]
            assert idx not in self.idx_to_mos, f"Duplicate index: {idx}"
            score_counts = row[2:11]
            mos = sum([x * (i+1) for i, x in enumerate(score_counts)]) / sum(score_counts)
            self.idx_to_mos[idx] = mos
    
    def _idx_is_valid(self, idx):
        img_exists = os.path.exists(os.path.join(self.images_dir, f"{idx}.jpg"))
        mos_exists = int(idx) in self.idx_to_mos
        return img_exists and mos_exists
    
    def _create_img_paths(self):
        splits_file = os.path.join(DATASET_PATHS['ava']['data_dir'], 'ava_split_yu_et_al.json')
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        train_ids = splits['train_id']
        num_val = int(VAL_SIZE * len(train_ids))
        val_ids = self.rng.choice(train_ids, size=num_val, replace=False)
        train_ids = [idx for idx in train_ids if idx not in val_ids]
        test_ids = splits['test_id']

        train_img_paths = [os.path.join('images', f"{idx}.jpg") for idx in train_ids]
        test_img_paths = [os.path.join('images', f"{idx}.jpg") for idx in test_ids]
        val_img_paths = [os.path.join('images', f"{idx}.jpg") for idx in val_ids]

        return train_img_paths, val_img_paths, test_img_paths
        
    def _create_img_paths_old(self):

        img_list_filenames = os.listdir(self.image_lists_dir)
        train_img_list_filenames = [os.path.join(self.image_lists_dir, x) for x in img_list_filenames if '_train.jpg' in x]
        test_img_list_filenames = [os.path.join(self.image_lists_dir, x) for x in img_list_filenames if '_test.jpg' in x]

        trainval_indices = []
        test_indices = []
        for img_list_filename in train_img_list_filenames:
            with open(img_list_filename, 'r') as f:
                reader = csv.reader(f)
                trainval_indices += [x[0] for x in list(reader) if self._idx_is_valid(x[0])]
        for img_list_filename in test_img_list_filenames:
            with open(img_list_filename, 'r') as f:
                reader = csv.reader(f)
                test_indices += [x[0] for x in list(reader) if self._idx_is_valid(x[0])]

        trainval_indices = list(set(trainval_indices))
        test_indices = list(set(test_indices))

        num_val = int(VAL_SIZE * len(trainval_indices))
        val_indices = self.rng.choice(trainval_indices, size=num_val, replace=False)
        train_indices = [idx for idx in trainval_indices if idx not in val_indices]

        train_img_paths = [os.path.join('images', f"{i}.jpg") for i in train_indices]
        val_img_paths = [os.path.join('images', f"{i}.jpg") for i in val_indices]
        test_img_paths = [os.path.join('images', f"{i}.jpg") for i in test_indices]

        return train_img_paths, val_img_paths, test_img_paths

    def _create_labels(self):

        labels = []
        for img_path in tqdm(self.img_paths):
            idx = img_path.split('/')[-1].split('.')[0]
            assert int(idx) in self.idx_to_mos, f"Index {idx} for img {img_path} not found in idx_to_mos"
            labels.append(self.idx_to_mos[int(idx)])
        
        if self.label_mode == 'regression':
            self.labels = labels
        elif self.label_mode == 'classification':
            self.labels = self._discretize_scores(labels, num_bins=5)
    

if __name__ == '__main__':
    dataset = AVA(
        attribute='good_looking_image',
        split='test',
        use_pairs=False,
    )
    dataset.visualize_examples()