import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from dotmap import DotMap
import pandas as pd
from statistics import mean
from flatten_dict import flatten
import wandb
import copy
from rankbench.utils import (
    ScoreNormalizer, 
    normalize_embeddings, 
)
from torch.utils.data import Subset

from rankbench.models import models as models_dict
from rankbench.constants import *
from rankbench.models.base import BaseExperiment    
from rankbench.models.losses import logistic_pairwise_loss, TASK_TO_LOSS
from rankbench.models.optim import CosineScheduleWithWarmup
from rankbench.utils import direction_from_text
from rankbench.models.base import EmbeddingsHandler

from rankbench.utils import make_experiment_deterministic

EQUAL_SUBSTRINGS_OK = [
    'num_batches_tracked',
]

EQUAL_KEYS_OK = []

class CrossDatasetEvalExperiment(BaseExperiment):

    def __init__(self, args, logger, wandb_run):
        super().__init__(args, logger, wandb_run)

        self.csv_url = args.csv_url
        url = self.csv_url.split("::")[0]
        worksheet = self.csv_url.split("::")[1]
        csv_data = download_csv(url, worksheet)
        self.df = pd.DataFrame(csv_data, columns=csv_data[0])
        self.wandb_api = wandb.Api()

        self._collect_eval_configs()

    def _collect_eval_configs(self):
        self.eval_configs = []

        datasets = self.df['delta:dataset.name'].unique()
        datasets = [d for d in datasets if not d in ['', 'delta:dataset.name']]
        self.datasets = datasets

        for dataset in datasets:
            subset_df = self.df[self.df['delta:dataset.name'] == dataset].sort_values(by='val.spearman', ascending=False)
            best_model = subset_df.iloc[0]
            wandb_url = best_model['WandB url']
            model_run_id = wandb_url.split('/')[-1]
            encoder_name = best_model['delta:model']
            run = self.wandb_api.run(f"mode-connect/rankanything/{model_run_id}")
            run_config = DotMap(run.config)
            downstream = self._create_downstream_model(
                model_type='linear',
                input_dim=512,
                output_dim=1
            )
            checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, f'{model_run_id}.pth'))
            downstream.load_state_dict(checkpoint['downstream'])

            self.eval_configs.append({
                'encoder_name': encoder_name,
                'downstream': downstream,
                'training_dataset': run_config.dataset.name,
                'normalize_embeddings': run_config.normalize_embeddings,
                'normalize_scores': run_config.normalize_scores,
                'task': run_config.task,
                'attribute': run_config.attribute,
            })

        additional_eval_configs = []
        for i in range(len(self.eval_configs)):
            for j in range(i+1, len(self.eval_configs)):
                avg_model = self._average_models([self.eval_configs[i]['downstream'], self.eval_configs[j]['downstream']])
                assert self.eval_configs[i]['normalize_embeddings'] == self.eval_configs[j]['normalize_embeddings']
                assert self.eval_configs[i]['normalize_scores'] == self.eval_configs[j]['normalize_scores']
                assert self.eval_configs[i]['task'] == self.eval_configs[j]['task']
                additional_eval_configs.append({
                    'encoder_name': self.eval_configs[i]['encoder_name'],
                    'downstream': avg_model,
                    'training_dataset': f"{self.eval_configs[i]['training_dataset']} - {self.eval_configs[j]['training_dataset']}",
                    'normalize_embeddings': self.eval_configs[i]['normalize_embeddings'],
                    'normalize_scores': self.eval_configs[i]['normalize_scores'],
                    'task': self.eval_configs[i]['task'],
                    'attribute': self.eval_configs[i]['attribute'],
                })
        
        self.eval_configs.extend(additional_eval_configs)

    def _evaluate_model_on_dataset(self, dataset, model_config):
        # get the metadata from the wandb url
        encoder = models_dict[model_config['encoder_name']]().cuda().eval()

        train_ds, val_ds, test_ds = self._load_dataset(
            dataset_config=DotMap({'name': dataset, 'settings': {'use_pairs': False}}),
            transform=encoder.preprocess,
            attribute=model_config['attribute'],
            task=model_config['task'],
        )

        embeddings_handler = EmbeddingsHandler(
            encoder=encoder,
            datasets={'train': train_ds, 'val': val_ds, 'test': test_ds},
            model_name=model_config['encoder_name'],
            normalize_embeddings=model_config['normalize_embeddings'],
            logger=self.logger,
            horizontal_flip=False,
            supports_scores=train_ds.supports_scores,
            use_pairs=False,
            train_shots=self.train_shots,
        )

        downstream = model_config['downstream']

        results = {}
        embeddings_dl = DataLoader(embeddings_handler.embedding_datasets['test'], batch_size=100, shuffle=False)
        results = self._evaluate(embeddings_dl, None, downstream, normalize_scores=model_config['normalize_scores'], task=model_config['task'])
        results['training_dataset'] = model_config['training_dataset']
        results['eval_dataset'] = dataset
        return results
    

    def _average_models(self, model_list):
        avg_model = copy.deepcopy(model_list[0])
        avg_state_dict = {}

        for model in model_list:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                if key not in avg_state_dict:
                    avg_state_dict[key] = state_dict[key]
                else:
                    avg_state_dict[key] += state_dict[key]
        avg_state_dict = {k: v / len(model_list) for k, v in avg_state_dict.items()}
        avg_model.load_state_dict(avg_state_dict)
        return avg_model
    
    def _evaluate(self, dl, encoder, downstream, normalize_scores=False, task=None):
        
        score_normalizer = None
        if normalize_scores:
            score_normalizer = ScoreNormalizer(scores=None)

        gt = []
        preds = []

        for X, y in dl:
            X = X.cuda()
            y = y.cuda()
            gt.append(y)
            with torch.no_grad():
                if encoder is not None:
                    X = encoder.encode_image(X)
                outputs = downstream(X).squeeze()
                preds.append(outputs)
        preds = torch.cat(preds, dim=0)
        if score_normalizer:
            preds = score_normalizer.unnormalize(preds)
        gt = torch.cat(gt, dim=0)
        results = self._create_results_from_scores(preds, gt, pairs=None, task=task)
        return results

    def __call__(self):

        self.results = []
        for dataset in self.datasets:
            for eval_config in self.eval_configs:

                self.results.append(self._evaluate_model_on_dataset(dataset, eval_config))
                self.logger.info(f"Results for {dataset} and model trained on {eval_config['training_dataset']}: {self.results[-1]}")
        
        # flatten the results
        columns = list(self.results[0].keys())
        table = wandb.Table(columns=columns)
        for result in self.results:
            table.add_data(*[result[key] for key in columns])
        self.wandb_run.log({'results': table})

class TrainingExperiment(BaseExperiment):

    def __init__(self, args, logger=None, wandb_run=None):
        super().__init__(args, logger, wandb_run)
        make_experiment_deterministic(args.random_seed)
        self.train_shots = args.train_shots if isinstance(args.train_shots, int) else None
        self.zero_shot_prompt = args.zero_shot_prompt
        self.normalize_embeddings = args.normalize_embeddings
        self.normalize_scores = args.normalize_scores
        self.dataset_config = args.dataset
        self.model_name = args.model
        self.scorer_name = args.scorer_name
        self.save_best_ckpt = args.save_best_ckpt
        self.use_pairs = args.use_pairs
        self.freeze_encoder = args.freeze_encoder
        self.attribute = args.attribute
        print(f"Attribute passed to training experiment: {self.attribute}")
        self.training_config = args.training

        if self.train_shots is not None and self.train_shots > 2:
            self.training_config.batch_size = min(self.training_config.batch_size, self.train_shots)
            self.logger.info(f"Batch size set to {self.training_config.batch_size} for train shots: {self.train_shots}")

        self.use_val_set_for_training = args.use_val_set_for_training

        assert args.task in ['regression', 'classification']
        self.task = args.task
        if self.task == 'classification' and self.normalize_scores:
            raise ValueError('Normalizing scores is not supported for classification tasks!')
        
        self.horizontal_flip = self.training_config.horizontal_flip

        assert args.downstream_model in ['linear', 'nonlinear', 'average_baseline', 'zero_shot', 'zero_shot_direction']
        self.downstream_model = args.downstream_model
        self.downstream_layers = args.downstream_layers

        self.logger.info(f"Loading model: {self.model_name}")
        self.pretrained_encoder = args.pretrained_encoder if args.pretrained_encoder is not None else True
        self.logger.info(f"Pretrained encoder arg inside training experiment: {self.pretrained_encoder}")
        print(f"Pretrained encoder arg inside training experiment: {self.pretrained_encoder}")
        self.model = models_dict[self.model_name](pretrained=self.pretrained_encoder).cuda().eval()
        self._sanity_check_model_weights()

        self._create_scorer()

        if self.dataset_config.name == 'adience':
            self.logger.info(f"Loading Adience dataset with test fold: {self.dataset_config['settings']['test_fold']}")

        train_ds, val_ds, test_ds = self._load_dataset(
            dataset_config=self.dataset_config,
            transform=self.model.preprocess,
            attribute=self.attribute,
            task=self.task,
        )

        if self.dataset_config.name == 'adience':
            self.logger.info(f"Loaded Adience dataset with test fold: {train_ds.test_fold} for train, {val_ds.test_fold} for val, {test_ds.test_fold} for test")

        print(f"Train shots passed to embeddings handler: {self.train_shots}")
        self.embeddings_handler = EmbeddingsHandler(
            encoder=self.model,
            use_cache_for_embeddings=self.use_cache_for_embeddings,
            datasets={'train': train_ds, 'val': val_ds, 'test': test_ds},
            model_name=self.model_name,
            normalize_embeddings=self.normalize_embeddings,
            logger=self.logger,
            horizontal_flip=self.horizontal_flip,
            use_val_set_for_training=self.use_val_set_for_training,
            supports_scores=test_ds.supports_scores,
            use_pairs=self.use_pairs,
            train_shots=self.train_shots,
        )
        if not self.dataset_config.name in ['lfw']:
            if not self.use_val_set_for_training:
                self._disjoint_datasets_sanity_check([train_ds.img_paths, val_ds.img_paths, test_ds.img_paths])
            else:
                self._disjoint_datasets_sanity_check([train_ds.img_paths, test_ds.img_paths])

        self.embedding_dim = self.embeddings_handler.embedding_dim

        if not self.freeze_encoder:
            
            self.train_ds = train_ds
            self.val_ds = val_ds
            self.test_ds = test_ds

        else:

            self.train_ds = self.embeddings_handler.embedding_datasets['train']
            self.val_ds = self.embeddings_handler.embedding_datasets['val']
            self.test_ds = self.embeddings_handler.embedding_datasets['test']

            if self.task == 'classification':
                self.train_ds.num_classes = train_ds.num_classes
                self.val_ds.num_classes = val_ds.num_classes
                self.test_ds.num_classes = test_ds.num_classes

        self.train_loader = DataLoader(self.train_ds, batch_size=self.training_config.batch_size, shuffle=True)
        self.train_loader_eval = DataLoader(self.train_ds, batch_size=self.training_config.batch_size, shuffle=False)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.training_config.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.training_config.batch_size, shuffle=False)

        self.downstream = self._create_downstream_model(
            model_type=self.downstream_model,
            input_dim=self.embedding_dim,
            output_dim=1 if self.task == 'regression' else self.train_ds.num_classes,
        )

        if self.normalize_scores:
            self.score_normalizer = ScoreNormalizer(self.train_ds.labels)

        if not self.use_pairs:
            if 'loss_fn' not in self.training_config:
                self.criterion = TASK_TO_LOSS[self.task]
            else:
                self.criterion = TASK_TO_LOSS[self.training_config['loss_fn']]
        else:
            self.criterion = logistic_pairwise_loss
        self.logger.info(f"Criterion: {self.criterion}")
        
        self._make_pairs_eval()

        self.logger.info(f"Train set has {(len(self.train_ds))} examples")
        self.logger.info(f"Val set has {(len(self.val_ds))} examples")
        self.logger.info(f"Test set has {(len(self.test_ds))} examples")


    def _sanity_check_model_weights(self):
        if not self.pretrained_encoder:
            dummy_model = models_dict[self.model_name](pretrained=True)
            sd1 = self.model.state_dict()
            sd2 = dummy_model.state_dict()
            for k in sd1.keys():
                if len([s for s in EQUAL_SUBSTRINGS_OK if s in k]) > 0:
                    continue
                assert not torch.allclose(sd1[k].cpu(), sd2[k].cpu()), f"Weights for {k} are the same"
        
    def _make_pairs_eval(self):
        self.train_pairs_eval = None
        self.val_pairs_eval = None
        self.test_pairs_eval = None
        if self.use_pairs and self.train_ds.supports_scores:
            self.train_pairs_eval = self._create_pairs_from_scores(self.train_ds.labels)
            self.val_pairs_eval = self._create_pairs_from_scores(self.val_ds.labels)
            self.test_pairs_eval = self._create_pairs_from_scores(self.test_ds.labels)

    def _disjoint_datasets_sanity_check(self, lists):
        for i in range(len(lists)):
            for j in range(i+1, len(lists)):
                assert not set(lists[i]).intersection(lists[j]), f"Lists {i} and {j} are not disjoint"

    def _average_regression_baseline(self, train_ds, eval_ds, use_noise=True):
        train_labels = [float(label) for label in train_ds.labels]
        eval_labels = [float(label) for label in eval_ds.labels]
        mean_train_label = mean(train_labels)
        preds = torch.tensor([mean_train_label] * len(eval_labels)).cuda().float()

        if use_noise:
            noise = torch.normal(0, 0.1, size=preds.shape).cuda()
            preds += noise
        return preds

    def _average_classification_baseline(self, train_ds, eval_ds):
        label_counts = {}
        for label in train_ds.labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        most_common_label = max(label_counts, key=label_counts.get)
        preds = torch.tensor([most_common_label] * len(eval_ds.labels)).cuda().long()
        return preds

    def _average_baseline_one_dataset(self, train_ds, eval_ds):
        if self.task == 'regression':
            preds = self._average_regression_baseline(train_ds, eval_ds)
        elif self.task == 'classification':
            preds = self._average_classification_baseline(train_ds, eval_ds)
        results = self._create_results_from_scores(preds, eval_ds.labels, pairs=None, task=self.task)
        return results

    def _average_baseline(self):
        results = {}
        for ds, split in zip([self.train_ds, self.val_ds, self.test_ds], ['train', 'val', 'test']):
            results[split] = self._average_baseline_one_dataset(self.train_ds, ds)
        return flatten(results, reducer='dot')

    def _create_scorer(self):
        if self.scorer_name is not None:
            self.scorer = models_dict[self.scorer_name](encoder=self.model).cuda()
        else:
            self.scorer = None

    def _zero_shot_eval(self, use_extremes=False):
        extremes_suffix = '_extreme' if use_extremes else ''
        prompts_filename = os.path.join(PROMPTS_DIR, f'{self.dataset_config.name}_{self.attribute}{extremes_suffix}.txt')
        with open(prompts_filename, 'r') as f:
            prompts = f.readlines()
            original_length = len(prompts)
            prompts = list(set(prompts))
            self.logger.info(f"Original length: {original_length}, after removing duplicates: {len(prompts)}")

            # try out all prompts and return the best one
            best_results = None
            best_prompt = None
            for prompt in tqdm(prompts):
                prompt = prompt.strip()
                if use_extremes:
                    prompt = prompt.split('$')
                    lower, upper = prompt[0], prompt[1]
                    direction = direction_from_text(lower, upper, self.model)
                    pred_scores_train = torch.matmul(self.embeddings_handler.embeddings['train'], direction).squeeze()
                    pred_scores_val = torch.matmul(self.embeddings_handler.embeddings['val'], direction).squeeze()
                    pred_scores_test = torch.matmul(self.embeddings_handler.embeddings['test'], direction).squeeze()
                else:
                    pred_scores_train = self.scorer(attribute=prompt, embeddings=self.embeddings_handler.embeddings['train'])['scores']
                    pred_scores_val = self.scorer(attribute=prompt, embeddings=self.embeddings_handler.embeddings['val'])['scores']
                    pred_scores_test = self.scorer(attribute=prompt, embeddings=self.embeddings_handler.embeddings['test'])['scores']
                results = {
                    'train': self._create_results_from_scores(pred_scores_train, self.train_ds.labels, self.train_pairs_eval, task=self.task),
                    'val': self._create_results_from_scores(pred_scores_val, self.val_ds.labels, self.val_pairs_eval, task=self.task),
                    'test': self._create_results_from_scores(pred_scores_test, self.test_ds.labels, self.test_pairs_eval, task=self.task),
                }
                if best_results is None:
                    best_results = results
                if results['val']['spearman'] > best_results['val']['spearman']:
                    best_results = results
                    best_prompt = prompt
        
        best_results['prompt'] = best_prompt
        self.logger.info(f"Best prompt: {best_prompt}")
        best_results = flatten(best_results, reducer='dot')
        for k, v in best_results.items():
            self.wandb_run.log({k: v})
        return best_results

    def _output_with_amp(self, X, y):
        raise NotImplementedError('This function is out of date!')
        with autocast():
            if not self.freeze_encoder:
                X = self.model.encode_image(X)
            if self.normalize_embeddings:
                X = self.embeddings_handler._normalize_embeddings(X)
            outputs = self.downstream(X).squeeze()
            loss = self.criterion(outputs, y)
        return outputs, loss
    
    def _output_without_amp(self, X, y):
        if not self.freeze_encoder:
            X = self.model.encode_image(X)
        outputs = self.downstream(X).squeeze()
        if self.task == 'classification':
            y = y.long()
        loss = self.criterion(outputs, y)
        return outputs, loss

    def _evaluate_pairwise_comparison(self, dl):

        correct = 0
        total = 0

        for X1, X2, y in dl:
            X1 = X1.cuda()
            X2 = X2.cuda()
            y = y.cuda()
            with torch.no_grad():
                if not self.freeze_encoder:
                    X1 = self.model.encode_image(X1)
                    X2 = self.model.encode_image(X2)
                    if self.normalize_embeddings:
                        X1 = normalize_embeddings(X1)
                        X2 = normalize_embeddings(X2)

                out1 = self.downstream(X1)
                out2 = self.downstream(X2)

                pred = (out1 > out2).squeeze()
                correct += (pred == y).sum().item()
                total += y.size(0)

        return {'accuracy': correct / total}
                
    def _evaluate(self, dl, pairs):

        gt = []
        preds = []
        losses = []

        for X, y in dl:
            X = X.cuda()
            y = y.cuda()
            gt.append(y)

            with torch.no_grad():
                outputs, loss = self._output_without_amp(X, y)
                if self.task == 'classification':
                    outputs = torch.argmax(outputs, dim=1)
                else:
                    outputs = outputs.squeeze()
                if X.shape[0] == 1: 
                    outputs = torch.tensor([outputs.item()]).cuda()
                preds.append(outputs)
                losses.append(loss.item())

        preds = torch.cat(preds, dim=0)
        if self.normalize_scores:
            preds = self.score_normalizer.unnormalize(preds)
        gt = torch.cat(gt, dim=0)
        results = self._create_results_from_scores(preds, gt, pairs, task=self.task)
        results['loss'] = mean(losses)
        return results

    def _create_optimizer(self, downstream):
        self.logger.info(f"Creating optimizer for downstream model: {downstream}")
        if self.freeze_encoder:
            optimizer = torch.optim.Adam(
                downstream.parameters(), 
                lr=self.training_config.downstream_lr,
                weight_decay=self.training_config.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW([
                {'params': downstream.parameters(), 'lr': self.training_config.downstream_lr},
                {'params': self.model.parameters(), 'lr': self.training_config.clip_lr}
            ], weight_decay=self.training_config.weight_decay)
        return optimizer

    def _backprop_pairs_without_amp(self, X1, X2, y, optimizer):

        if not self.freeze_encoder:
            X1 = self.model.encode_image(X1)
            X2 = self.model.encode_image(X2)
            if self.normalize_embeddings:
                X1 = normalize_embeddings(X1)
                X2 = normalize_embeddings(X2)

        out1 = self.downstream(X1)
        out2 = self.downstream(X2)
        loss = self.criterion(out1, out2, y)
        loss.backward()
        if not self.freeze_encoder:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def _backprop_without_amp(self, X, y, optimizer):
        outputs, loss = self._output_without_amp(X, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
    
    def _backprop_with_amp(self, X, y, optimizer):
        outputs, loss = self._output_with_amp(X, y)
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        return loss.item()
                
    def _train(self):
        
        if not self.freeze_encoder:
            self.model = self.model.float()
        self.logger.info(f"Downstream model: {self.downstream}")

        optimizer = self._create_optimizer(self.downstream)
        self.logger.info(f"Optimizer: {optimizer}")

        steps = len(self.train_loader) * self.training_config.epochs

        if not self.freeze_encoder:
            scheduler = CosineScheduleWithWarmup(optimizer, steps, steps * 0.1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)

        losses = []
        
        if not self.use_pairs:
            if self.task == 'classification':
                val_criterion = 'accuracy'
            else:
                val_criterion = 'spearman'
        else:
            val_criterion = 'accuracy'
            
        best_val_criterion = -float('inf')
        best_results = None
        step = 0
        best_ckpt = None
        for epoch in tqdm(range(self.training_config.epochs)):

            if not self.freeze_encoder:
                self.model.unfreeze_image_encoder()
                self.model.train()
            self.downstream.train()
            self.downstream.requires_grad_(True)

            for sample in self.train_loader:
                
                if self.use_pairs:
                    X1, X2, y = sample
                    X1 = X1.cuda()
                    X2 = X2.cuda()
                else:
                    X, y = sample
                    X = X.cuda()

                y = y.cuda()

                if self.normalize_scores:
                    y = self.score_normalizer.normalize(y)
                if self.task == 'classification':
                    y = y.long()
                else:
                    y = y.float()
                optimizer.zero_grad()
                
                if not self.freeze_encoder and self.use_amp:
                    loss = self._backprop_with_amp(X, y, optimizer)
                elif self.use_pairs:
                    loss = self._backprop_pairs_without_amp(X1, X2, y, optimizer)
                else:
                    loss = self._backprop_without_amp(X, y, optimizer)

                losses.append(loss)
                scheduler.step(step)
                step += 1
                lr_data = {
                    'step': step,
                    'downstream_lr': optimizer.param_groups[0]['lr'],
                }
                if len(optimizer.param_groups) > 1:
                    lr_data['encoder_lr'] = optimizer.param_groups[1]['lr']
                self.wandb_run.log(lr_data)

            if not self.freeze_encoder:
                self.model.freeze_image_encoder()
                self.model.eval()

            if not self.use_pairs:
                epoch_results = {
                    'epoch': epoch,
                    'train': self._evaluate(dl=self.train_loader_eval, pairs=self.train_pairs_eval),
                    'val': self._evaluate(dl=self.val_loader, pairs=self.val_pairs_eval),
                    'test': self._evaluate(dl=self.test_loader, pairs=self.test_pairs_eval),
                }
            else:
                epoch_results = {
                    'epoch': epoch,
                    'train': self._evaluate_pairwise_comparison(dl=self.train_loader_eval),
                    'val': self._evaluate_pairwise_comparison(dl=self.val_loader),
                    'test': self._evaluate_pairwise_comparison(dl=self.test_loader),
                }
            epoch_results = flatten(epoch_results, reducer='dot')
            if epoch_results[f'val.{val_criterion}'] > best_val_criterion:
                best_val_criterion = epoch_results[f'val.{val_criterion}']
                best_results = epoch_results
                
                best_ckpt = {'downstream': self.downstream.state_dict()}
                if not self.freeze_encoder:
                    best_ckpt['encoder'] = self.model.state_dict()
            for k, v in epoch_results.items():
                self.wandb_run.log({k: v})

        for k, v in best_results.items():
            self.wandb_run.summary[k] = v
            
        if self.save_best_ckpt:
            self._save_best_ckpt(best_ckpt)
        return best_results

    def _save_best_ckpt(self, best_ckpt):
        run_id = self.wandb_run.id
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f'{run_id}.pth')
        torch.save(best_ckpt, ckpt_path)

    def _create_pairs_from_scores(self, scores):
        pairs = []
        self.logger.info(f"Creating pairs from scores.")
        for i in tqdm(range(len(scores))):
            for j in range(i+1, len(scores)):
                if scores[i] == scores[j]:
                    continue
                elif scores[i] > scores[j]:
                    pairs.append((i, j, 1))
                else:
                    pairs.append((i, j, 0))
        return pairs

    def __call__(self):
        if self.downstream_model == 'average_baseline':
            self.results = self._average_baseline()
        elif self.downstream_model == 'zero_shot':
            self.results = self._zero_shot_eval(use_extremes=False)
        elif self.downstream_model == 'zero_shot_direction':
            self.results = self._zero_shot_eval(use_extremes=True)
        else:
            self.results = self._train()

    def print_results(self):
        self.logger.info(f"Results: {self.results}")