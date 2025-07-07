import os
from tqdm import tqdm

from rankbench.datasets.base import PairwiseComparisonDataset

class PalmFruitRipeness(PairwiseComparisonDataset):

    def __init__(
        self,
        images_dir,
        attribute='decay',
        transform=None,
        dataset_name='palm_fruit_ripeness',
        split='train'
    ):
        assert attribute == 'decay', f"Invalid attribute: {attribute}"
        self.classes = ['0Immature', '1PartiallyRipe', '2FullyRipe', '3OverRipe', '4Decayed']
        self.class_to_idx = {
            'Immature': 0,
            'PartiallyRipe': 1,
            'FullyRipe': 2,
            'OverRipe': 3,
            'Decayed': 4
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

        if self.split == 'train':
            image_names_list_filename = 'Training.txt'
        elif self.split == 'test':
            image_names_list_filename = 'Testing.txt'
        else:
            image_names_list_filename = 'Validation.txt'
        
        with open(os.path.join(self.images_dir, 'Train_val_test_split', image_names_list_filename), 'r') as f:
            img_names = f.readlines()
            img_names = [n.strip() for n in img_names]

        self.img_paths = []
        self.labels = []

        for image_name in img_names:
            for c in self.class_to_idx.keys():
                if c in image_name:
                    label = self.class_to_idx[c]
                    self.labels.append(label)
                    img_path = os.path.join(self.images_dir, 'Images', f'{label}{c}', image_name)
                    assert os.path.exists(img_path)
                    assert img_path.endswith('.jpg')
                    self.img_paths.append(img_path)
                    break
    
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
    dataset = PalmFruitRipeness(
        images_dir=DATASET_PATHS['ripeness']['images_dir'],
        attribute='decay', split='test'
    )
    dataset.visualize_examples()
    dataset.visualize_pairs()