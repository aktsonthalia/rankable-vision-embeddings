import logging
import wandb
import yaml
from dotmap import DotMap
import json
import numpy as np
from torch.utils.data import Subset
from argparse import ArgumentParser

from stuned.utility.helpers_for_main import prepare_wrapper_for_experiment
from stuned.utility.logger import try_to_log_in_csv_in_batch

from rankbench.models.experiments import TrainingExperiment, CrossDatasetEvalExperiment
from rankbench.utils import get_git_file_commit_url


def check_config_for_training_experiment(config, config_path, logger):
    pass

def main(config, logger, processes_to_kill_before_exiting):

    args = DotMap(config)

    # add git commit hash to config
    git_commit_url = get_git_file_commit_url(__file__)
    if git_commit_url:
        config['git_commit_url'] = git_commit_url
        logger.info(f"Git commit URL: {git_commit_url}")

    wandb.login()

    try:
        wandb_run = logger.wandb_run
    except AttributeError:
        wandb_run = wandb.init(
            entity=args.logging.entity,
            project=args.logging.project,
            config=config,
        )
        logger.wandb_run = wandb_run
    
    if args.experiment_type == 'training':
        experiment = TrainingExperiment(
            args=args,
            logger=logger,
            wandb_run=wandb_run,
        )   

        experiment()

        logger.info(f"Results: {experiment.results}")
        metrics = ['accuracy', 'tau', 'spearman', 'mae', 'loss']

        results_tuple_to_log = []
        for metric in metrics:
            for split in ['train', 'val', 'test']:
                key = f'{split}.{metric}'
                if key in experiment.results:
                    results_tuple_to_log.append((key, experiment.results[key]))
    
        for key, value in results_tuple_to_log:
            wandb_run.summary[f'best_{key}'] = value

        if args.using_csv:
            logger.info(f"Logging results in csv")
            try_to_log_in_csv_in_batch(logger, results_tuple_to_log)
        else:
            # print the results nested dict in a pretty way
            print(json.dumps(experiment.results, indent=4))
    
    elif args.experiment_type == 'cross_dataset_eval':
        experiment = CrossDatasetEvalExperiment(
            args=args,
            logger=logger,
            wandb_run=wandb_run,
        )
        experiment()

    wandb_run.finish()

if __name__ == '__main__':
    print(f"Running experiment")
    parser = ArgumentParser()
    parser.add_argument('--do_not_use_csv', '-d', action='store_true', default=False)
    parser.add_argument('--config_path', '-c', type=str, required=False)
    args = parser.parse_args()

    if args.do_not_use_csv:
        with open(args.config_path, 'r') as f:
            config = DotMap(yaml.safe_load(f))
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        main(config, logger, [])
    else:
        prepare_wrapper_for_experiment(
            check_config_for_training_experiment
        )(main)()
