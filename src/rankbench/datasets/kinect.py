# https://github.com/natanielruiz/deep-head-pose/blob/master/code/datasets.py

import os
from tqdm import tqdm
import numpy as np

from rankbench.datasets.base import PairwiseComparisonDataset

class Kinect(PairwiseComparisonDataset):

    def __init__(
        self, 
        attribute='yaw', 
        transform=None, 
        supports_scores=True,
        split='train',
        dataset_name='kinect',
        use_pairs=False,
        label_mode='regression',
        logger=None,
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
        )


    def _extract_img_paths_from_seqs(self, seqs):
        img_paths = []
        for seq in seqs:
            seq_dir = os.path.join(self.data_dir, 'faces_0', seq)
            img_paths.extend([os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if f.endswith('_rgb.png')])
        return img_paths

    def _get_pose_matrix(self, img_path):
        # https://github.com/natanielruiz/deep-head-pose/blob/f7bbb9981c2953c2eca67748d6492a64c8243946/code/datasets.py#L524C9-L533C28
        pose_path = img_path.replace('_rgb.png', '_pose.txt')
        with open(pose_path, 'r') as f:
            R = []
            for line in f:
                line = line.strip('\n').split(' ')
                l = []
                if line[0] != '':
                    for nb in line:
                        if nb == '':
                            continue
                        l.append(float(nb))
                    R.append(l)

        R = np.array(R)
        R = R[:3, :]
        R = np.transpose(R)
        return R
    
    def _get_euler_angles(self, R, key):
        # https://github.com/natanielruiz/deep-head-pose/blob/f7bbb9981c2953c2eca67748d6492a64c8243946/code/datasets.py#L542

        if key == 'yaw':
            value = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        elif key == 'pitch':
            value = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
        elif key == 'roll':
            value = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi

        return value

    def _create_img_paths(self):
        
        train_seqs = ['01', '02', '04', '07', '08', '09', '13', '14', '16', '19', '20', '21', '22', '23', '24']
        val_seqs = ['11']
        test_seqs = ['03', '05', '06', '10', '12', '15', '17', '18']

        train_img_paths = self._extract_img_paths_from_seqs(train_seqs)
        val_img_paths = self._extract_img_paths_from_seqs(val_seqs)
        test_img_paths = self._extract_img_paths_from_seqs(test_seqs)

        return train_img_paths, val_img_paths, test_img_paths

    def _create_labels(self):
        labels = []
        for img_path in self.img_paths:
            pose_matrix = self._get_pose_matrix(img_path)   
            label = self._get_euler_angles(pose_matrix, self.attribute)
            labels.append(label)
        
        if self.label_mode == 'regression':
            self.labels = labels
            # check if there are any nan values
            if np.isnan(labels).any():
                raise ValueError(f"Found nan values in labels for attribute: {self.attribute}")
                
        elif self.label_mode == 'classification':
            min_label = min(labels)
            max_label = max(labels)
            # each bin should be 3 degrees wide
            num_bins = int((max_label - min_label) / 3)
            self.labels = self._discretize_scores(labels, num_bins=num_bins)

if __name__ == '__main__':
    dataset = Kinect(
        attribute='roll'
    )
    dataset.visualize_examples(n=8)