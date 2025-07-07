from functools import partial
from copy import deepcopy
from torch.cuda.amp import GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score

from rankbench.datasets import datasets
from rankbench.datasets.base import EmbeddingDataset
from rankbench.utils import (
    normalize_embeddings, 
    extract_image_embeddings_or_load_from_cache, 
    compute_metrics, 
    pairwise_comparison_accuracy, 
)


class BaseExperiment:

    def __init__(self, args, logger=None, wandb_run=None):
        self.logger = logger
        self.wandb_run = wandb_run
        self.use_cache_for_embeddings = args.use_cache_for_embeddings
        self.use_amp = args.use_amp
    
    def _prepare_amp(self):
        assert not self.use_amp, f"We are not using AMP for now"
        if not self.freeze_encoder and self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
    def _load_dataset(self, dataset_config, transform, attribute, task):
        dataset_constructor = partial(
            datasets[dataset_config.name], 
            attribute=attribute,    
            transform=transform,
            label_mode=task,
            **dataset_config.settings
        )
        train_ds = dataset_constructor(split='train')
        val_ds = dataset_constructor(split='val')
        test_ds = dataset_constructor(split='test')
        return train_ds, val_ds, test_ds

    def _create_downstream_model(self, model_type, input_dim, output_dim):

        if model_type == 'linear':
            return nn.Linear(input_dim, output_dim).cuda()
        elif model_type == 'nonlinear':
            downstream = nn.Sequential()
            downstream.append(nn.Linear(input_dim, 128))
            downstream.append(nn.ReLU())
            for _ in range(self.downstream_layers-2):
                downstream.append(nn.Linear(128, 128))
                downstream.append(nn.ReLU())
            downstream.append(nn.Linear(128, output_dim))
            return downstream.cuda()
    
    def _create_results_from_scores(self, preds, gt, pairs, task):

        results = {}

        if gt is not None:
            if not isinstance(preds, np.ndarray):
                if isinstance(preds, torch.Tensor):
                    preds = preds.detach().cpu().numpy()
                elif isinstance(preds, list):
                    preds = np.array(preds)
            if not isinstance(gt, np.ndarray):
                if isinstance(gt, torch.Tensor):
                    gt = gt.detach().cpu().numpy()
                elif isinstance(gt, list):
                    gt = np.array(gt)

        if task == 'regression':
            results = compute_metrics(preds, gt)
            if pairs is not None:
                results['pairwise_comparison_accuracy'] = pairwise_comparison_accuracy(preds, pairs)

        elif task == 'classification':
            results['accuracy'] = accuracy_score(preds, gt)

        return results

class EmbeddingsHandler:
    def __init__(
        self, 
        encoder, 
        use_cache_for_embeddings=True, 
        datasets=None, 
        model_name=None, 
        normalize_embeddings=True,
        logger=None,
        use_val_set_for_training=False,
        horizontal_flip=False,
        supports_scores=True,
        use_pairs=False,
        train_shots=None,
    ):
        self.train_shots = train_shots
        self.encoder = encoder
        self.use_pairs = use_pairs
        self.use_cache_for_embeddings = use_cache_for_embeddings
        self.datasets = datasets
        self.supports_scores = supports_scores
        if self.supports_scores:
            self.labels = {k: torch.tensor(v.labels) for k, v in self.datasets.items()}
        else:
            self.labels = {k: None for k in self.datasets}
        
        if self.use_pairs:
            self.pairs = {k: v.pairs for k, v in self.datasets.items()}
        else:
            self.pairs = {k: None for k in self.datasets}
        self.flipped_datasets = {k: deepcopy(v.image_ds) for k, v in self.datasets.items()}
        for k in self.flipped_datasets:
            self.flipped_datasets[k].horizontal_flip = True
        self.model_name = model_name
        self.image_embeddings_cache_path = os.path.join(self.datasets['train'].embeddings_dir, f'{self.model_name}.pt')
        self.normalize_embeddings = normalize_embeddings
        self.logger = logger
        self.use_val_set_for_training = use_val_set_for_training
        self.horizontal_flip = horizontal_flip
        self._load_embeddings()
        self._create_embedding_datasets()
    
    def _load_embeddings(self):
        self.logger.info(f"Extracting image embeddings.") 

        extract_embeddings_fn = partial(
            extract_image_embeddings_or_load_from_cache, 
            model=self.encoder,
            use_cache=self.use_cache_for_embeddings,
        )
        image_dl = partial(DataLoader, batch_size=100, shuffle=False)
        
        self.embeddings = {
            'train': extract_embeddings_fn(dl=image_dl(self.datasets['train'].image_ds), cache_path=self.image_embeddings_cache_path.replace('.pt', '_train.pt')),
            'val': extract_embeddings_fn(dl=image_dl(self.datasets['val'].image_ds), cache_path=self.image_embeddings_cache_path.replace('.pt', '_val.pt')),
            'test': extract_embeddings_fn(dl=image_dl(self.datasets['test'].image_ds), cache_path=self.image_embeddings_cache_path.replace('.pt', '_test.pt')),
            'train_flipped': None,
            'val_flipped': None,
        }
        if self.horizontal_flip:
            self.embeddings['train_flipped'] = extract_embeddings_fn(dl=image_dl(self.flipped_datasets['train']), cache_path=self.image_embeddings_cache_path.replace('.pt', '_train_flipped.pt'))
            self.embeddings['val_flipped'] = extract_embeddings_fn(dl=image_dl(self.flipped_datasets['val']), cache_path=self.image_embeddings_cache_path.replace('.pt', '_val_flipped.pt'))
        for k in self.embeddings:
            if self.embeddings[k] is None:
                continue
            self.embeddings[k] = self.embeddings[k].float()
            if self.normalize_embeddings:
                self.embeddings[k] = normalize_embeddings(self.embeddings[k])

        self.embedding_dim = self.embeddings['train'].shape[1]

        if self.train_shots is not None:
            indices = np.random.choice(len(self.embeddings['train']), size=self.train_shots, replace=False)
            self.embeddings['train'] = self.embeddings['train'][indices]
            self.labels['train'] = self.labels['train'][indices]
            if self.horizontal_flip:
                self.embeddings['train_flipped'] = self.embeddings['train_flipped'][indices]
    
    def _merge_train_and_val(self):
        self.embeddings['train'] = torch.cat([self.embeddings['train'], self.embeddings['val']], dim=0)
        self.labels['train'] = torch.cat([self.labels['train'], self.labels['val']], dim=0)
        if self.horizontal_flip:
            self.embeddings['train_flipped'] = torch.cat([self.embeddings['train_flipped'], self.embeddings['val_flipped']], dim=0)

    def _create_embedding_datasets(self):
        if self.use_val_set_for_training:
            self._merge_train_and_val()
        self.embedding_datasets = {
            'train': EmbeddingDataset(self.embeddings['train'], self.labels['train'], self.embeddings['train_flipped'], self.pairs['train']),
            'val': EmbeddingDataset(self.embeddings['val'], self.labels['val'], self.embeddings['val_flipped'], self.pairs['val']),
            'test': EmbeddingDataset(self.embeddings['test'], self.labels['test'], pairs=self.pairs['test']),
        }
        for k in self.embedding_datasets:
            self.embedding_datasets[k].supports_scores = self.supports_scores