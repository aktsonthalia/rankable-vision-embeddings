import os
from tqdm import tqdm
from scipy.io import loadmat

from rankbench.datasets.base import PairwiseComparisonDataset

class Emotic(PairwiseComparisonDataset):

    def __init__(self, images_dir, attribute='valence', transform=None, labels_file=None):
        self.labels_file = labels_file
        self.img_dirs = ['ade20k', 'emodb_small', 'framesdb', 'mscoco']
        super().__init__(images_dir, transform, attribute)

    def _create_img_paths(self):
        self.img_paths = []
        self.labels = []

        label_data = loadmat(self.labels_file)
        breakpoint()
        for row in label_data.iterrows():
            name = row[1]['Name']
            img_path = os.path.join(self.images_dir, f'{name}.png')
            self.img_paths.append(img_path)
            assert img_path.endswith('.png')
            assert os.path.exists(img_path)
            label = row[1]['Subjective Score']
            self.labels.append(label)


if __name__ == '__main__':
    from rankbench.constants import DATASET_PATHS
    dataset = SlideFocus(
        images_dir=DATASET_PATHS['slidefocus']['images_dir'],
        labels_file=DATASET_PATHS['slidefocus']['labels_file'],
        attribute='focus',
    )
    dataset.visualize_examples()