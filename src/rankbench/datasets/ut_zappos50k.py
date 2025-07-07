from rankbench.datasets.base import PairwiseComparisonDataset
import os 
from scipy.io import loadmat


class UTZap50kDataset(PairwiseComparisonDataset):

    def __init__(self, images_dir, attribute='open', transform=None, labels_file=None, image_paths_file=None):
        
        self.labels_file = labels_file
        assert os.path.exists(self.labels_file)
        self.image_paths_file = image_paths_file
        assert os.path.exists(self.image_paths_file)
        self.attribute_to_idx = {
            'open': 1, 
            'pointy': 2,
            'sporty': 3,
            'comfort': 4,
        }
        self.attribute_idx = self.attribute_to_idx[attribute]
        super().__init__(images_dir, transform, attribute)
    
    def _create_img_paths(self):
        image_paths = loadmat(self.image_paths_file, simplify_cells=True)
        self.img_paths = image_paths['imagepath']
        self.img_paths = [os.path.join(self.images_dir, p) for p in self.img_paths]
        for img_path in self.img_paths: 
            assert os.path.exists(img_path)

    def _make_pairwise_comparisons(self):
        labels = loadmat(self.labels_file, simplify_cells=True)
        self.labels = labels['mturkOrder'][self.attribute_idx - 1]
        self.pairs = []

        for pair in self.labels:
            idx1 = int(pair[0]) - 1
            idx2 = int(pair[1]) - 1
            assert os.path.exists(self.img_paths[idx1])
            assert os.path.exists(self.img_paths[idx2])
            assert pair[2] == self.attribute_idx
            assert pair[3] in {1, 2}
            if pair[3] == 1: label = 1
            if pair[3] == 2: label = 0
            self.pairs.append((idx1, idx2, label))

        print(f"Generated {len(self.pairs)} pairwise comparisons")

if __name__ == '__main__':
    from rankbench.constants import DATASET_PATHS
    dataset = UTZap50kDataset(
        images_dir=DATASET_PATHS['ut_zap50k']['images_dir'],
        labels_file=DATASET_PATHS['ut_zap50k']['labels_file'],
        image_paths_file=DATASET_PATHS['ut_zap50k']['image_paths_file'],
        attribute='pointy'
    )
    dataset.visualize_examples()