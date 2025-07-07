import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from functools import partial

from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import VAL_SIZE


class Adience(PairwiseComparisonDataset):

    def __init__(
        self, 
        attribute='age', 
        transform=None, 
        supports_scores=True,
        split='train',
        dataset_name='adience',
        test_fold=0,
        use_pairs=False,
        label_mode='regression',
        logger=None,
        use_val_set_for_training=False,
    ):
        self.test_fold = test_fold
        self.age_groups = {
            '(0, 2)': 1,
            '(4, 6)': 2,
            # '(8, 12)': 3,
            '(8, 13)': 3,
            '(15, 20)': 4,
            '(25, 32)': 5,
            # '(27, 32)': 5,
            # '(38, 42)': 6,
            '(38, 43)': 6,
            # '(38, 48)': 6,
            '(48, 53)': 7,
            '(60, 100)': 8
        }
        self.age_groups_to_fix = {
            '35': 6, '3': 1, '55': 8, '58': 8, 
            '22': 4, '13': 3, '45': 6, '36': 6, 
            '23': 5, '57': 8, '56': 7, '2': 1, 
            '29': 5, '34': 5, '42': 6, '46': 7,
            '32': 5, '(38, 48)': 6, '(38, 42)': 6,
            '(8, 23)': 3, '(27, 32)': 5,
            '(8, 12)': 3
        }

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
        os.makedirs(os.path.join(self.data_dir, "cached_outputs", f"fold_{self.test_fold}_is_test"), exist_ok=True)
        return os.path.join(self.data_dir, "cached_outputs", f"fold_{self.test_fold}_is_test", f"{split}_img_paths.txt")
    
    def _make_embeddings_dir_path(self):
        return os.path.join(self.data_dir, "cached_outputs", f"fold_{self.test_fold}_is_test", 'embeddings')
    
    def _get_labels_df(self, fold_files):
        labels_df = None
        for fold_file in fold_files:
            with open(fold_file, 'r') as f:
                df = pd.read_csv(f, sep='\t')
                if labels_df is None:
                    labels_df = df
                else:
                    labels_df = pd.concat([labels_df, df])
        return labels_df

    def _get_img_paths_to_labels(self, labels_df):
        img_paths_to_labels = {}
        for i, row in labels_df.iterrows():
            if not isinstance(row['age'], str) and np.isnan(row['age']):
                continue
            img_path = os.path.join(
                'aligned', 
                row['user_id'],
                f"landmark_aligned_face.{row['face_id']}.{row['original_image']}"
            )
            age = row['age']
            label = self.age_groups[age] if age in self.age_groups else self.age_groups_to_fix[age]
            img_paths_to_labels[img_path] = label
        return img_paths_to_labels

    def _create_img_paths(self):
        
        train_folds_files = [os.path.join(self.data_dir, f"fold_{i}_data.txt") for i in range(5) if not i == self.test_fold]
        test_folds_files = [os.path.join(self.data_dir, f"fold_{self.test_fold}_data.txt")]

        labels_df_trainval = self._get_labels_df(train_folds_files)
        labels_df_trainval = labels_df_trainval.reset_index(drop=True)
        labels_df_test = self._get_labels_df(test_folds_files)

        # check if labels_df_trainval contains duplicates
        if labels_df_trainval.duplicated().any():
            raise ValueError("labels_df_trainval contains duplicates")

        # find unique user_ids in labels_df_trainval
        unique_user_ids_trainval = labels_df_trainval['user_id'].unique()
        self.rng.shuffle(unique_user_ids_trainval)
        num_val = int(VAL_SIZE * len(unique_user_ids_trainval))
        user_ids_train = unique_user_ids_trainval[:-num_val]
        user_ids_val = unique_user_ids_trainval[-num_val:]

        labels_df_train = labels_df_trainval[labels_df_trainval['user_id'].isin(user_ids_train)]
        labels_df_val = labels_df_trainval[labels_df_trainval['user_id'].isin(user_ids_val)]

        # reset indices
        labels_df_train = labels_df_train.reset_index(drop=True)
        labels_df_val = labels_df_val.reset_index(drop=True)
        labels_df_test = labels_df_test.reset_index(drop=True)

        self.img_paths_to_labels = {
            'train': self._get_img_paths_to_labels(labels_df_train),
            'val': self._get_img_paths_to_labels(labels_df_val),
            'test': self._get_img_paths_to_labels(labels_df_test),
        }

        train_img_paths = list(self.img_paths_to_labels['train'].keys())
        val_img_paths = list(self.img_paths_to_labels['val'].keys())
        test_img_paths = list(self.img_paths_to_labels['test'].keys())

        return train_img_paths, val_img_paths, test_img_paths

    def _create_labels(self):
        folds_files = [os.path.join(self.data_dir, f"fold_{i}_data.txt") for i in range(5)]
        labels_df = self._get_labels_df(folds_files)
        img_paths_to_labels = self._get_img_paths_to_labels(labels_df)
        img_paths = {os.path.join(self.data_dir, k): v for k, v in img_paths_to_labels.items()}
        self.labels = [img_paths[img_path] for img_path in self.img_paths]

        if self.label_mode == 'classification':
            self.labels = [x-1 for x in self.labels]

if __name__ == '__main__':
    dataset_partial = partial(
        Adience,
        attribute='age',
        use_pairs=False,
        split='val',
        label_mode='classification',
    )
    val_ds = dataset_partial(split='val')
    train_ds = dataset_partial(split='train')
    test_ds = dataset_partial(split='test')
    total_images = len(train_ds.image_ds) + len(test_ds.image_ds) + len(val_ds.image_ds)
    print(f"Total images: {total_images}")
    print(f"Train images: {len(train_ds.image_ds)}")
    print(f"Val images: {len(val_ds.image_ds)}")
    print(f"Test images: {len(test_ds.image_ds)}")
    test_ds.visualize_examples(n=8)