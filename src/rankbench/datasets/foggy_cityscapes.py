from tqdm import tqdm
from rankbench.datasets.base import PairwiseComparisonDataset
import os

class FoggyCityscapes(PairwiseComparisonDataset):

    def __init__(self, images_dir, attribute='fog', transform=None):
        super().__init__(images_dir, transform, attribute)
    
    def _create_img_paths(self):
        self.img_paths = []
        for root, _, files in os.walk(self.images_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
                    self.img_paths.append(os.path.join(root, f))
        for img_path in self.img_paths:
            assert os.path.exists(img_path)
        
    def _make_pairwise_comparisons(self):

        N = len(self.img_paths)
        self.pairs = []
        self.ties = []

        for i in tqdm(range(N)):
            for j in range(i+1, N):
                fog_level_i = float(self.img_paths[i].split('.png')[0].split('_')[-1])
                fog_level_j = float(self.img_paths[j].split('.png')[0].split('_')[-1])
                if fog_level_i < fog_level_j:
                    label = 0
                elif fog_level_i > fog_level_j:
                    label = 1
                else:
                    self.ties.append((i, j))
                    continue
                
                self.pairs.append((i, j, label))

        print(f"Generated {len(self.pairs)} pairwise comparisons")
        print(f"Detected {len(self.ties)} ties")
    

if __name__ == '__main__':
    from rankbench.constants import DATASET_PATHS
    dataset = FoggyCityscapes(images_dir=DATASET_PATHS['foggy_cityscapes']['images_dir'])
    dataset.visualize_examples()