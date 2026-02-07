import torch
import argparse
import os
import yaml
import logging
import wandb
import pickle
from collections import defaultdict
import numpy as np
import json

from data_preprocessing.data_loader import prepare_kfold_datasets, load_fold_data, combine_train_val
from models.deepcov_lstm import DeepConvLSTM
from models.transformer import TransformerClassifier
from models.mlstm import MLSTM_FCN  
from training.trainer import train_model
from evals.evaluation import evaluate_model, evaluate_early_classification
from utils.logger import setup_logging
from utils.wandb import manage_wandb_logs
from utils.summary import print_kfold_summary, save_kfold_summary_to_csv
from torch.utils.data import DataLoader, ConcatDataset

def load_yaml_config(dataset_name):
    config_path = os.path.join('configs', f'{dataset_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_args():
    parser = argparse.ArgumentParser()
    # --- General Arguments ---
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--dataset', type=str, default='doore')
    parser.add_argument('--padding', type=str, default='mean')
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--aug_method', type=str, default='noise')  

    # --- Model Arguments ---
    parser.add_argument('--model_type', type=str, default='transformer', choices=['deepconvlstm', 'transformer', 'mlstm'])
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--ff_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'cls'])
    parser.add_argument('--use_cls_token', type=bool, default=False)
    parser.add_argument('--use_batchnorm', type=bool, default=True)
    parser.add_argument('--classifier_hidden', type=int, default=64)

    # --- Training Arguments ---
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--prepare_data', action='store_true', help='Whether to generate and save k-fold data')
    parser.add_argument('--use_saved_config', action='store_true',
                    help='Use previously saved best config for training instead of grid search')


    # --- Parse and Merge ---
    args = parser.parse_args()
    dataset_config = load_yaml_config(args.dataset)
    for key, value in dataset_config.items():
        setattr(args, key, value)

    # --- Paths ---
    args.save_dir = getattr(args, 'save_dir', 'save_model')
    os.makedirs(args.save_dir, exist_ok=True)

    args.save_name = getattr(args, 'save_name', f"{args.dataset}_{args.model_type}.pth")
    args.save_path = os.path.join(args.save_dir, args.save_name)

    args.cm_image_path = getattr(args, 'cm_image_path', f"results/{args.dataset}_{args.model_type}_cm.png")
    os.makedirs(os.path.dirname(args.cm_image_path), exist_ok=True)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def setup_model(args, input_channels, num_classes):
    if args.model_type == 'deepconvlstm':
        return DeepConvLSTM(input_channels=input_channels, num_classes=num_classes)
    elif args.model_type == 'transformer':
        return TransformerClassifier(
            input_dim=input_channels,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            pooling=args.pooling,
            use_cls_token=args.use_cls_token,
            use_batchnorm=args.use_batchnorm,
            classifier_hidden=args.classifier_hidden
        )
    elif args.model_type == 'mlstm':        
        return MLSTM_FCN(input_channels=input_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


def main():
    torch.cuda.empty_cache()
    args = setup_args()

    # -------------------- Logging & W&B Setup --------------------
    manage_wandb_logs(wandb_dir='wandb', max_runs=5)
    log_path = setup_logging(base_log_dir='logs', dataset_name=args.dataset, mode=args.mode, max_logs=5)
    logging.info(f"Using device: {args.device}")

    # -------------------- Data Preparation -----------------------
    if args.prepare_data or not os.path.exists(f"fold_data/{args.dataset}/fold_0.pkl"):
        label_map, input_channels = prepare_kfold_datasets(args)
    else:
        with open(f"fold_data/{args.dataset}/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        label_map = meta["label_map"]
        input_channels = meta["input_channels"]

    # -------------------- Result Accumulators --------------------
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
    early_acc_by_step = defaultdict(list)
    early_acc_by_fold = []
    early_classwise_acc_by_fold = []

    # -------------------- K-Fold Loop ----------------------------
    for fold_id in range(args.k_fold):
        logging.info(f"\n=== Fold {fold_id} ===")
        fold_path = f"fold_data/{args.dataset}/fold_{fold_id}.pkl"
        train_loader, val_loader, test_loader = load_fold_data(fold_path, args, mode='train')

        wandb.init(project="EarlyHAR", name=f"{args.dataset}_fold{fold_id}", config=vars(args))

        if args.mode == 'train':
            logging.info("Starting training...")

            # ===== Case 1: Use pre-saved best config =====
            if args.use_saved_config:
                config_path = os.path.join("checkpoints", args.dataset, f"fold{fold_id}_config.json")
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found: {config_path}")
                with open(config_path, "r") as f:
                    best_config = json.load(f)

            # ===== Case 2: Run grid search =====
            else:
                logging.info("Starting hyperparameter search...")
                best_model = None
                best_val_score = -np.inf
                best_config = {}

                for hidden_dim in [64, 128, 256]:
                    for lr in [1e-5, 3e-4, 1e-3, 1e-2]:
                        args.model_dim = hidden_dim
                        args.learning_rate = lr
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                        model = setup_model(args, input_channels, num_classes=len(label_map)).to(args.device)
                        val_score = train_model(model, train_loader, val_loader, args, return_val_score=True)

                        if val_score > best_val_score:
                            best_val_score = val_score
                            best_model = model
                            best_config = {'model_dim': hidden_dim, 'learning_rate': lr}

                logging.info(f"[Best Config] model_dim={best_config['model_dim']}, learning_rate={best_config['learning_rate']}")
                
                # save the best config
                ckpt_dir = os.path.join("checkpoints", args.dataset)
                os.makedirs(ckpt_dir, exist_ok=True)
                with open(os.path.join(ckpt_dir, f"fold{fold_id}_config.json"), "w") as f:
                    json.dump(best_config, f)
            
        
            # ----- Retrain regardless of config source -----
            args.model_dim = best_config['model_dim']
            args.learning_rate = best_config['learning_rate']
            logging.info(f"[Retrain] model_dim={args.model_dim}, learning_rate={args.learning_rate}")
            
            trainval_loader = combine_train_val(train_loader, val_loader, batch_size=args.batch_size)
            final_model = setup_model(args, input_channels, num_classes=len(label_map)).to(args.device)
            logging.info(f"[Check] final_model.input_proj.weight.shape = {final_model.input_proj.weight.shape}")

            train_model(final_model, trainval_loader, val_loader=None, args=args)
            torch.save(final_model.state_dict(), os.path.join("checkpoints", args.dataset, f"fold{fold_id}.pt"))

        elif args.mode == 'test':
            logging.info("Starting evaluation...")

            # ----- Load best config -----
            config_path = os.path.join("checkpoints", args.dataset, f"fold{fold_id}_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    best_config = json.load(f)
                    for k, v in best_config.items():
                        setattr(args, k, v)
            else:
                logging.warning(f"Best config file not found: {config_path}")

            # ----- Load model -----
            model = setup_model(args, input_channels, num_classes=len(label_map)).to(args.device)
            logging.info(f"[Check] final_model.input_proj.weight.shape = {model.input_proj.weight.shape}")
            model_path = os.path.join("checkpoints", args.dataset, f"fold{fold_id}.pt")
            model.load_state_dict(torch.load(model_path))

            # ----- Test Evaluation -----
            results = evaluate_model(model, test_loader, args)
            wandb.log({
                "Test Accuracy": results['accuracy'],
                "Test Precision": results['precision'],
                "Test Recall": results['recall'],
                "Test F1": results['f1_score']
            })

            for metric in fold_metrics:
                fold_metrics[metric].append(results[metric])

            # ----- Early Classification Evaluation -----
            early_results, classwise_results = evaluate_early_classification(model, test_loader, args)
            early_acc_by_fold.append(early_results)
            early_classwise_acc_by_fold.append(classwise_results)

            for step, acc in early_results.items():
                logging.info(f"Early Accuracy at {int(step * 100)}% sequence: {acc:.4f}")
                early_acc_by_step[step].append(acc)

        wandb.finish()
        torch.cuda.empty_cache()

    # -------------------- Final Summary -------------------------
    if args.mode == 'test':
        logging.info("=== Final Evaluation Summary ===")
        print_kfold_summary(fold_metrics, early_acc_by_step)
        best_hm, mean_best_hm = save_kfold_summary_to_csv(
            args.dataset,
            fold_metrics,
            early_acc_by_step,
            early_acc_by_fold,
            early_classwise_acc_by_fold  
        )
        logging.info(f"Best HM: {best_hm}")
        logging.info(f"Mean Best HM: {mean_best_hm}")

if __name__ == '__main__':
    main()
