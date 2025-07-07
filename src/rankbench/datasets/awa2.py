import os
from tqdm import tqdm
import csv
from PIL import Image
import numpy as np

from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import DATASET_PATHS, VAL_SIZE


class AwA2(PairwiseComparisonDataset):

    def __init__(
            self, 
            attribute='black', 
            transform=None, 
            supports_scores=True,
            split='train',
            dataset_name='awa2',
            use_pairs=False,
            label_mode='regression',
            logger=None,
            use_val_set_for_training=False,
        ):
        self.images_dir = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'Animals_with_Attributes2', 'JPEGImages')
        self.classes_file = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'Animals_with_Attributes2', 'classes.txt')
        self.train_classes_file = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'Animals_with_Attributes2', 'trainclasses.txt')
        self.test_classes_file = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'Animals_with_Attributes2', 'testclasses.txt')
        self.predicates_names_file = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'Animals_with_Attributes2', 'predicates.txt')
        self.predicates_values_file = os.path.join(DATASET_PATHS[dataset_name]['data_dir'], 'Animals_with_Attributes2', 'predicate-matrix-continuous.txt')
        self._create_classes_to_predicates()

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
    
    def _create_classes_to_predicates(self):
        with open(self.predicates_values_file, 'r') as f:
            predicates_values = np.array([list(map(float, x.strip().split())) for x in f.readlines()])
            
        # Convert each column (predicate) to dense ranks
        for j in range(predicates_values.shape[1]):
            column = predicates_values[:, j]
            # Get unique values in sorted order
            unique_values = np.unique(column)
            # Create rank mapping (lowest value gets rank 0)
            rank_map = {val: rank for rank, val in enumerate(unique_values)}
            # Apply ranking
            predicates_values[:, j] = np.array([rank_map[val] for val in column])

        with open(self.predicates_names_file, 'r') as f:
            predicates_names = [x.strip().split('\t')[1] for x in f.readlines()]
        
        assert len(predicates_values) == 50
        assert len(predicates_values[0]) == 85
        
        with open(self.classes_file, 'r') as f:
            classes = [x.strip().split('\t')[1] for x in f.readlines()]
        assert len(classes) == 50
        
        self.classes_to_predicates = {}

        for i, cls in enumerate(classes):
            self.classes_to_predicates[cls] = {}
            for j, predicate in enumerate(predicates_names):
                self.classes_to_predicates[cls][predicate] = predicates_values[i][j]
    
    def _class_names_to_img_filenames(self, class_names):
        img_filenames = []
        for class_name in class_names:
            class_img_filenames = os.listdir(os.path.join(self.images_dir, class_name))
            class_img_filenames = [os.path.join('Animals_with_Attributes2', 'JPEGImages', class_name, img_filename) for img_filename in class_img_filenames]
            img_filenames.extend(class_img_filenames)
        
        return img_filenames
    
    def _create_img_paths(self):

        with open(self.train_classes_file, 'r') as f:
            trainval_classes = [x.strip() for x in f.readlines()]
        with open(self.test_classes_file, 'r') as f:
            test_classes = [x.strip() for x in f.readlines()]

        self.rng.shuffle(trainval_classes)
        train_classes = trainval_classes[:-5]
        val_classes = trainval_classes[-5:]

        train_img_paths = self._class_names_to_img_filenames(train_classes)
        val_img_paths = self._class_names_to_img_filenames(val_classes)
        test_img_paths = self._class_names_to_img_filenames(test_classes)

        return train_img_paths, val_img_paths, test_img_paths

    def _create_labels(self):

        labels = []
        for img_path in tqdm(self.img_paths):
            class_name = img_path.split('/')[-2]
            predicate_value = self.classes_to_predicates[class_name][self.attribute]
            labels.append(predicate_value)
        
        if self.label_mode == 'regression':
            self.labels = labels
        elif self.label_mode == 'classification':
            self.labels = self._discretize_scores(labels, num_bins=10)
    

if __name__ == '__main__':
    dataset = AwA2(
        attribute='coastal',
        split='test',
        use_pairs=False,
    )
    dataset.visualize_examples(n=8)