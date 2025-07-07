import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import ImageFile
import seaborn as sns
ImageFile.LOAD_TRUNCATED_IMAGES = True

from rankbench.constants import DATASET_SPLIT_SEED, DATASET_PATHS


class PairwiseComparisonDataset(Dataset):

    def __init__(
            self,
            dataset_name,
            transform=None, 
            attribute=None, 
            supports_scores=False,
            split='train',
            use_pairs=False,
            label_mode='regression',
            logger=None,
            use_val_set_for_training=False,
        ):
        self.use_val_set_for_training = use_val_set_for_training
        self.rng = np.random.default_rng(DATASET_SPLIT_SEED)
        self.dataset_name = dataset_name
        self._make_data_dir()
        self.embeddings_dir = self._make_embeddings_dir_path()
        assert os.path.exists(self.data_dir), f"Invalid path: {self.data_dir}"
        self.attribute = attribute
        self.transform = transform
        self.supports_scores = supports_scores
        assert split in ['train', 'test', 'val'], f"Invalid split: {split}"
        self.split = split
        assert label_mode in ['regression', 'classification'], f"Invalid label mode: {label_mode}"
        self.label_mode = label_mode
        self._load_img_paths()
        self.use_pairs = use_pairs
        if self.supports_scores:
            self._create_labels()
            assert self.labels is not None, "Labels must be provided if dataset supports scores."

        if self.label_mode == 'classification':
            self.num_classes = len(set(self.labels))
        self.logger = logger
        if self.use_pairs:
            self._make_pairwise_comparisons()

        self.image_ds = ImageDataset(
            self.img_paths, 
            transform=self.transform, 
            labels=None if not self.supports_scores else self.labels
        )

    def _make_data_dir(self):
        self.data_dir = DATASET_PATHS[self.dataset_name]['data_dir']

    def _make_embeddings_dir_path(self):
        return os.path.join(self.data_dir, 'embeddings')

    def has_scores(self):
        return self.supports_scores
    
    def _create_img_paths(self):
        raise NotImplementedError

    def _create_labels(self):
        if self.supports_scores:
            raise NotImplementedError
        return
    
    def _make_img_paths_filename(self, split):
        return os.path.join(self.data_dir, f"{split}_img_paths.txt")
    
    def _save_img_paths(self, img_paths, split):
        with open(self._make_img_paths_filename(split), 'w') as f:
            for img_path in img_paths:
                f.write(f"{img_path}\n")
    
    def _load_img_paths(self):
        img_paths_filename = self._make_img_paths_filename(self.split)

        if not os.path.exists(img_paths_filename):
            train_paths, val_paths, test_paths = self._create_img_paths()
            self._save_img_paths(train_paths, 'train')
            self._save_img_paths(val_paths, 'val')
            self._save_img_paths(test_paths, 'test')
        
        with open(img_paths_filename, 'r') as f:
            img_paths = f.readlines()
            img_paths = [x.strip() for x in img_paths]
            img_paths = [os.path.join(self.data_dir, x) for x in img_paths]
            self.img_paths = img_paths
        

        if self.split == 'train' and self.use_val_set_for_training:

            val_img_paths_filename = self._make_img_paths_filename('val')
            with open(val_img_paths_filename, 'r') as f:
                val_img_paths = f.readlines()
                val_img_paths = [x.strip() for x in val_img_paths]
                val_img_paths = [os.path.join(self.data_dir, x) for x in val_img_paths]
                self.img_paths.extend(val_img_paths)
                

    def __len__(self):
        if self.use_pairs:
            return len(self.pairs)
        else:
            return len(self.image_ds)

    def __getitem__(self, idx):
        if self.use_pairs:
            idx1, idx2, label = self.pairs[idx]
            img1 = self.image_ds[idx1]
            img2 = self.image_ds[idx2]
            return img1, img2, label
        else:
            img, label = self.image_ds[idx]
            return img, label

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

    def _discretize_scores(self, scores, num_bins=10):
        labels = []
        min_score = min(scores)
        max_score = max(scores)
        bins = np.linspace(min_score, max_score, num=num_bins+1)

        for score in scores:
            label = np.digitize(score, bins, right=True) - 1
            if label == -1:
                label = 0
            assert label in range(num_bins), f"Invalid label: {label}"
            labels.append(label)

        return labels
    
    def sanity_check_all_images_exist(self):
        for img_path in self.img_paths:
            assert os.path.exists(img_path), f"Image {img_path} does not exist"
    
    def visualize_examples(self, n=5, randomize=True):
        self.image_ds.visualize_examples(n, randomize, save_path=f'examples/{self.dataset_name}_{self.split}_{self.attribute}_examples.png', attribute=self.attribute)

    def _make_histogram(self, save_path='dataset_histogram.png'):

        # make kde plot of the labels
        sns.kdeplot(self.labels)
        plt.title(f'{self.dataset_name} {self.split} histogram')
        plt.savefig(save_path)

    def visualize_pairs(self, n=5):
        idxs = np.random.choice(len(self), n, replace=False)
        fig, axes = plt.subplots(2, n, figsize=(20, 4))
        for i, idx in enumerate(idxs):
            img1_idx, img2_idx, label = self[idx]
            img1 = Image.open(self.img_paths[img1_idx])
            img2 = Image.open(self.img_paths[img2_idx])
            axes[0, i].imshow(img1)
            if label == 0:
                axes[0, i].set_title("Top < Bottom")
            elif label == 1:
                axes[0, i].set_title("Top > Bottom")
            axes[0, i].axis('off')
            axes[1, i].imshow(img2)
            axes[1, i].axis('off')
        print('Saving pairwise examples to pairwise_examples.png')
        plt.savefig('pairwise_examples.png')

class ImageDataset(Dataset):

    def __init__(self, img_paths, transform=None, labels=None, horizontal_flip=False):
        self.img_paths = img_paths
        self.transform = transform
        self.labels = labels
        self.horizontal_flip = horizontal_flip

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        if self.horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.transform:
            img = img.convert('RGB') # TODO make sure output does not change for earlier models like openaiclip if we convert to RGB before transform
            img = self.transform(img)
        if self.labels:
            return img, self.labels[idx]
        else:
            return img
    
    def visualize_examples(self, n=5, randomize=True, save_path='dataset_examples.png', attribute=None):
        if randomize:
            idxs = np.random.choice(len(self), n, replace=False)
        else:
            idxs = range(n)
        fig, axes = plt.subplots(2, n//2, figsize=(20, 4))
        axes = axes.flatten()
        for i, idx in enumerate(idxs):
            img = Image.open(self.img_paths[idx])
            if self.horizontal_flip:    
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            axes[i].imshow(img)
            axes[i].set_title(f"{attribute}: {self.labels[idx]:.2f}")
            axes[i].axis('off')
        
        plt.savefig(save_path)

class EmbeddingDataset(Dataset):

    def __init__(self, embeddings, labels, flipped_embeddings=None, pairs=None, shots=None):
        self.embeddings = embeddings
        self.labels = labels
        self.flipped_embeddings = flipped_embeddings
        self.pairs = pairs
        self.shots = shots
        if self.flipped_embeddings is not None:
            assert not torch.equal(self.embeddings, self.flipped_embeddings), "Embeddings are equal"

    def __len__(self):
        if self.pairs is None:
            return len(self.embeddings)
        else:
            return len(self.pairs)
    
    def __getitem__(self, idx):

        if self.pairs is not None:
            img1_idx, img2_idx, label = self.pairs[idx]
            embedding1 = self.embeddings[img1_idx]
            embedding2 = self.embeddings[img2_idx]
            return embedding1, embedding2, label
        else:
            label = self.labels[idx]    
            if self.flipped_embeddings is None:
                embedding = self.embeddings[idx]
            else:
                toss = np.random.rand()
                embedding = self.embeddings[idx] if toss < 0.5 else self.flipped_embeddings[idx]
            return embedding, label
            
