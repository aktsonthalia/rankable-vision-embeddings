import os
import subprocess
import numpy as np
import torch

from datetime import datetime
from scipy.stats import spearmanr, kendalltau, pearsonr
from tqdm import tqdm
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import random

from rankbench.constants import RESULTS_DIR

def normalize_embeddings(embeddings):
    return embeddings / torch.norm(embeddings, dim=1, keepdim=True)

class ScoreNormalizer:
    def __init__(self, scores):
        return None

    def normalize(self, scores):
        return torch.log1p(scores)
    
    def unnormalize(self, scores):
        return torch.expm1(scores)

def extract_image_embeddings_or_load_from_cache(dl, model, cache_path, use_cache=True):
    if os.path.exists(cache_path) and use_cache:
        embeddings = torch.load(cache_path, weights_only=True)
    else:
        # print warning in red saying use_cache is True but cache_path does not exist
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dl):
                if len(batch) == 2:
                    batch, _ = batch
                batch = batch.cuda()
                embeddings.append(model.encode_image(batch).cpu().detach())
        embeddings = torch.cat(embeddings, dim=0)
        if use_cache:
            cache_dir = os.path.dirname(cache_path)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save(embeddings, cache_path)
    return embeddings

def evaluate_pairwise(dataloader, comparator):
    correct = 0
    total = 0
    for i, (image1, image2, prompt, label) in enumerate(tqdm(dataloader)):
        sim_diff = comparator(image1, image2, prompt)

        # if sim1 > sim2 and label == 0 (i.e., Left image ranks higher), then we are correct
        # if sim1 < sim2 and label == 1 (i.e., Right image ranks higher), then we are also correct
        # so we need logical XOR to check if we are correct
        label = label.cuda()
        correct += torch.logical_xor(sim_diff > 0, label).sum().item()
        total += len(label)

    return correct / total

def pairwise_comparison_accuracy(scores, pairs):
    total_pairs = len(pairs)
    pairs = torch.tensor(pairs).cuda()
    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]
    labels = pairs[:, 2]
    scores = torch.tensor(scores).cuda()
    diffs = scores[i_indices] - scores[j_indices]
    correct = torch.sum((diffs < 0) & (labels == 0) | (diffs > 0) & (labels == 1))
    acc = correct.item() / total_pairs
    return acc

def compute_metrics(x, y):

    tau, tau_p_value = kendalltau(x, y, variant='b', alternative='two-sided')
    spearman, spearman_p_value = spearmanr(x, y, alternative='two-sided')
    pearson, pearson_p_value = pearsonr(x, y, alternative='two-sided')
    # mean absolute error
    mae = np.mean(np.abs(x - y))

    return {
        'tau': float(tau),
        'tau_p_value': float(tau_p_value),
        'spearman': float(spearman),
        'spearman_p_value': float(spearman_p_value),
        'pearson': float(pearson),
        'pearson_p_value': float(pearson_p_value),
        'mae': float(mae)
    }

def get_git_file_commit_url(file_path: str) -> str:
    """
    Generates an HTTPS link to the current Git commit for a given file.
    
    Parameters:
        file_path (str): The relative or absolute path of the file within the repository.

    Returns:
        str: The URL to the file in the current commit.
    """
    try:
        # Get the latest commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

        # Get the repository URL (convert SSH to HTTPS if needed)
        repo_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).strip().decode("utf-8")

        # Convert SSH Git URL (git@github.com:user/repo.git) to HTTPS (https://github.com/user/repo)
        if repo_url.startswith("git@"):
            repo_url = repo_url.replace(":", "/").replace("git@", "https://")
        repo_url = repo_url.replace(".git", "")  # Remove .git at the end if present

        # Get the repository root directory
        repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")

        # Convert absolute path to relative path (if needed)
        abs_file_path = os.path.abspath(file_path)
        rel_file_path = os.path.relpath(abs_file_path, repo_root)

        # Construct commit URL for the file
        commit_url = f"{repo_url}/blob/{commit_hash}/{rel_file_path}"
        return commit_url

    except subprocess.CalledProcessError:
        return "Error: Unable to retrieve commit hash, repository URL, or file path. Ensure you're in a Git repository."
    
import numpy as np

def custom_kfold(n_samples, n_splits=5, train_fraction=0.1, shuffle=True, random_state=42):
    """
    Perform k-fold cross-validation where each training set contains only a small fraction of the data.
    
    Parameters:
    - n_samples: Number of samples in the dataset
    - n_splits: Number of folds 
    - train_fraction: Fraction of data to use for training (e.g., 0.1 for 10%)
    - shuffle: Whether to shuffle the data before splitting
    - random_state: Random seed for reproducibility
    
    Yields:
    - train_idx: Indices for training set (10% of total data)
    - test_idx: Indices for test set (remaining 90% of total data)
    """
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)
    
    fold_size = int(n_samples * train_fraction)  # 10% of total data
    
    for i in range(n_splits):
        # Randomly sample 10% for training
        train_idx = np.random.choice(indices, size=fold_size, replace=False)
        # The rest becomes the test set
        test_idx = np.setdiff1d(indices, train_idx, assume_unique=True)
        
        yield train_idx, test_idx

def create_results_file_path(results_filename):
    return os.path.join(RESULTS_DIR, f'{results_filename}_{datetime.now().strftime("%d_%b_%H_%M")}.json')

import gspread
import os
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

def download_google_sheet_worksheet(sheet_url: str, worksheet_name: str) -> pd.DataFrame:
    """
    Downloads a specific worksheet from a Google Sheet given its URL and worksheet name.
    
    Args:
        sheet_url (str): The base URL of the Google Sheet.
        worksheet_name (str): The name of the worksheet to fetch.
        
    Returns:
        pd.DataFrame: The contents of the worksheet as a pandas DataFrame.
    """
    creds_path = os.path.join(os.environ['HOME'], 'config', 'gauth', 'credentials.json')
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path)
    client = gspread.authorize(creds)

    sheet = client.open_by_url(sheet_url).worksheet(worksheet_name)
    data = sheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    
    return df

def direction_from_text(text1, text2, model):
    text1_embedding = model.encode_text(model.tokenize(text1))
    text2_embedding = model.encode_text(model.tokenize(text2))
    direction = text2_embedding - text1_embedding
    direction = direction / direction.norm()
    return direction.detach().cpu().T.float()

def upload_df_to_google_sheet(df: pd.DataFrame, sheet_url: str, worksheet_name: str):
    """
    Uploads a pandas DataFrame to a specific worksheet in a Google Sheet.
    
    Args:
        df (pd.DataFrame): The DataFrame to upload.
        sheet_url (str): The base URL of the Google Sheet.
        worksheet_name (str): The name of the worksheet to upload the DataFrame to.
    """
    creds_path = os.path.join(os.environ['HOME'], 'config', 'gauth', 'credentials.json')
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path)
    client = gspread.authorize(creds)

    sheet = client.open_by_url(sheet_url).worksheet(worksheet_name)
    sheet.update([df.columns.tolist()] + df.values.tolist())

def create_results_from_scores(preds, gt, pairs, task):

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

def evaluate(dl, model, pairs):

    gt = []
    preds = []
    losses = []

    for X, y in dl:
        X = X.cuda()
        y = y.cuda()
        gt.append(y)

        with torch.no_grad():
            outputs = model(X)
            outputs = outputs.squeeze()
            preds.append(outputs)

    preds = torch.cat(preds, dim=0)
    gt = torch.cat(gt, dim=0)
    results = create_results_from_scores(preds, gt, pairs, 'regression')
    return results

from rankbench.constants import CHECKPOINTS_DIR
def extract_state_dict_from_wandb_run(wandb_run_url_or_id):
    if wandb_run_url_or_id.startswith('https://wandb.ai/'):
        run_id = wandb_run_url_or_id.split('/')[-1]
    else:
        run_id = wandb_run_url_or_id
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{run_id}.pth')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist")
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def make_experiment_deterministic(seed: int = 0):
    """Set seed for reproducibility across common Python ML libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior in cuDNN (at the cost of performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Also set environment variable for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())