import os

from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import VAL_SIZE
import json

class HCI(PairwiseComparisonDataset):

    def __init__(
            self, 
            attribute='clicked_recently', 
            transform=None, 
            supports_scores=True,
            split='train',
            dataset_name='hci',
            use_pairs=True,
            label_mode='regression',
            logger=None,
            use_val_set_for_training=False,
        ):
        self.problematic_img_names = [
            'decade_e2cd1d87080d2525125f3bc6d320e302_duplicate.jpg',
        ]
        self.decade_to_label = {
            '1930s': 0,
            '1940s': 1,
            '1950s': 2,
            '1960s': 3,
            '1970s': 4,
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

    def _create_img_paths(self):
        train_img_paths = []
        val_img_paths = []
        test_img_paths = []

        decades = sorted(self.decade_to_label.keys())
        decade_dirs = [os.path.join(self.data_dir, decade) for decade in decades]

        with open(os.path.join(self.data_dir, 'hci_split_yu_et_al.json'), 'r') as f:
            split_data = json.load(f)

        trainval_ids = split_data['train_id']
        num_val = int(VAL_SIZE * len(trainval_ids))
        val_ids = self.rng.choice(trainval_ids, size=num_val, replace=False)
        train_ids = [i for i in trainval_ids if i not in val_ids]
        test_ids = split_data['test_id']

        for idx in train_ids:
            # find which decade the image belongs to
            found = False
            for decade_dir in decade_dirs:
                if f"{idx}.jpg" in os.listdir(decade_dir):
                    train_img_paths.append(os.path.join(decade_dir, f"{idx}.jpg"))
                    found = True
                    break
            if not found:
                raise ValueError(f"Image {idx} not found in any decade directory")

        for idx in val_ids:
            found = False
            for decade_dir in decade_dirs:
                if f"{idx}.jpg" in os.listdir(decade_dir):
                    val_img_paths.append(os.path.join(decade_dir, f"{idx}.jpg"))
                    found = True
                    break
            if not found:
                raise ValueError(f"Image {idx} not found in any decade directory")

        for idx in test_ids:
            found = False
            for decade_dir in decade_dirs:
                if f"{idx}.jpg" in os.listdir(decade_dir):
                    test_img_paths.append(os.path.join(decade_dir, f"{idx}.jpg"))
                    found = True
                    break
            if not found:
                raise ValueError(f"Image {idx} not found in any decade directory")

        return train_img_paths, val_img_paths, test_img_paths

    def _create_img_paths_old(self):
        
        decades = sorted(self.decade_to_label.keys())
        train_img_paths = []
        val_img_paths = []
        test_img_paths = []

        for decade in decades:
            decade_dir = os.path.join(self.data_dir, decade)
            if not os.path.exists(decade_dir):
                raise ValueError(f"Invalid path: {decade_dir}")
            img_names = [name for name in os.listdir(decade_dir) if name.endswith('.jpg') and name not in self.problematic_img_names]
            # randomly choose 50 indices
            test_indices = self.rng.choice(len(img_names), size=50, replace=False)
            train_indices = [i for i in range(len(img_names)) if i not in test_indices]
            num_val = int(VAL_SIZE * len(train_indices))
            val_indices = self.rng.choice(train_indices, size=num_val, replace=False)
            train_indices = [i for i in train_indices if i not in val_indices]

            for i in train_indices:
                train_img_paths.append(os.path.join(decade, img_names[i]))
            for i in val_indices:
                val_img_paths.append(os.path.join(decade, img_names[i]))
            for i in test_indices:
                test_img_paths.append(os.path.join(decade, img_names[i]))

        return train_img_paths, val_img_paths, test_img_paths

    def _create_labels(self):
        self.labels = []
        for img_path in self.img_paths:
            decade = img_path.split('/')[-2]
            self.labels.append(self.decade_to_label[decade])

if __name__ == '__main__':
    train_ds = HCI(split='train')
    val_ds = HCI(split='val')
    test_ds = HCI(split='test')

    print(f"Train dataset size: {len(train_ds.img_paths)}")
    print(f"Val dataset size: {len(val_ds.img_paths)}")
    print(f"Test dataset size: {len(test_ds.img_paths)}")
    train_ds.visualize_examples()
    train_ds.visualize_pairs()