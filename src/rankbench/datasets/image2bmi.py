import os
from tqdm import tqdm

from rankbench.datasets.base import PairwiseComparisonDataset

class Image2BMIDataset(PairwiseComparisonDataset):

    def __init__(
        self, 
        images_dir, 
        attribute='age', 
        transform=None, 
        split='train',
        dataset_name='2dimage2bmi'
    ):

        self.dir_to_use = os.path.join(images_dir, f'Image_{split}')
        self.attribute_to_idx = {
            'age': 2,
            'height': 3,
            'weight': 4
        }
        super().__init__(
            images_dir=images_dir,
            attribute=attribute,
            transform=transform,
            dataset_name=dataset_name,
            supports_scores=True,
            split=split
        )

    def _create_img_paths(self):
        self.img_paths = []
        self.img_paths = os.listdir(self.dir_to_use)
        self.img_paths = [os.path.join(self.dir_to_use, f) for f in self.img_paths]
        self.img_paths = [f for f in self.img_paths if int(f.split('/')[-1].split('_')[2]) >= 18]

        self.labels = []
        idx = self.attribute_to_idx[self.attribute]
        for img_path in self.img_paths:
            assert img_path.endswith('.png') or img_path.endswith('.jpg')
            assert os.path.exists(img_path)
            items = img_path.split('/')[-1].split('.')[0].split('_')
            if self.attribute == 'weight': 
                if items[idx] == '11430528]':
                    items[idx] = '11430528'
                if items[idx] == '8300740 (2)':
                    items[idx] = '8300740'
                if items[idx] == '5397750 (2)':
                    items[idx] = '5397750'
            label = int(items[idx])

            if self.attribute == 'height':
                label = label // 10**3
            
            if self.attribute == 'weight':
                label = label // 10**5
                
            self.labels.append(label)                
    
    def _make_pairwise_comparisons(self):

        self.pairs = []
        self.ties = []

        N = len(self.img_paths)
        for i in tqdm(range(N)):
            for j in range(i+1, N):
                if self.labels[i] < self.labels[j]:
                    label = 0
                elif self.labels[i] > self.labels[j]:
                    label = 1
                else:
                    self.ties.append((i, j))
                    continue
                self.pairs.append((i, j, label))
        
        print(f"Generated {len(self.pairs)} pairwise comparisons")
        print(f"Detected {len(self.ties)} ties")
        print(f"Total: {len(self.pairs)} comparisons")


if __name__ == '__main__':
    from rankbench.constants import DATASET_PATHS
    from functools import partial
    dataset = partial(
        Image2BMIDataset,
        images_dir=DATASET_PATHS['2dimage2bmi']['images_dir'],
    )
    dataset(split='test', attribute='weight').visualize_pairs()