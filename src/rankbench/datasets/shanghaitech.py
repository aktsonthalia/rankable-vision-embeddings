import os
from tqdm import tqdm
import scipy.io

from rankbench.datasets.base import PairwiseComparisonDataset
from rankbench.constants import DATASET_PATHS, VAL_SIZE


class ShanghaiTech(PairwiseComparisonDataset):

    def __init__(
        self, 
        attribute='crowd_count',
        part='A', 
        transform=None, 
        supports_scores=True,
        split='train',
        dataset_name='shanghaitech',
        use_pairs=False,
        label_mode='regression',
        logger=None,
        use_val_set_for_training=False,
    ):
        self.part = part
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

    def _make_data_dir(self):
        self.data_dir = os.path.join(DATASET_PATHS[self.dataset_name]['data_dir'], f'part_{self.part}_final')

    def _create_img_paths(self):

        trainval_dir = os.path.join(self.data_dir,  'train_data', 'images')
        test_dir = os.path.join(self.data_dir, 'test_data', 'images')

        trainval_img_paths = [os.path.join(self.data_dir, 'train_data', 'images', x) for x in os.listdir(trainval_dir)]
        self.rng.shuffle(trainval_img_paths)
        num_val = int(VAL_SIZE * len(trainval_img_paths))
        train_img_paths = trainval_img_paths[:-num_val]
        val_img_paths = trainval_img_paths[-num_val:]
        
        test_img_paths = [os.path.join(self.data_dir, 'test_data', 'images', x) for x in os.listdir(test_dir)]

        return train_img_paths, val_img_paths, test_img_paths

    def _create_labels(self):

        labels_split = 'train' if self.split in ['train', 'val'] else 'test'
        labels_dir = os.path.join(self.data_dir, f'{labels_split}_data', 'ground_truth')
        labels = []
        for img_path in tqdm(self.img_paths):
            img_name = img_path.split('/')[-1].split('.')[0]
            gt_mat_file_name = f'GT_{img_name}.mat'
            gt_mat_path = os.path.join(labels_dir, gt_mat_file_name)
            gt_mat = scipy.io.loadmat(gt_mat_path, simplify_cells=True)
            gt_count = gt_mat['image_info']['number']
            labels.append(gt_count)
        
        if self.label_mode == 'regression':
            self.labels = labels
        elif self.label_mode == 'classification':
            self.labels = self._discretize_scores(labels, num_bins=5)
    

if __name__ == '__main__':
    dataset = ShanghaiTech(
        attribute='crowd_count',
        part='B',
        split='train',
        use_pairs=False,
    )
    dataset._make_histogram(f'{dataset.split}_histogram.png')
    dataset.visualize_examples(n=8)