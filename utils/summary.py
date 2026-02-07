import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import logging

def get_label_names(dataset_name):
    label_mappings = {
        "doore": {
            0: "Small talk",
            1: "Studying together",
            2: "Technical discussion",
            3: "Seminar",
        },
        "opportunity": {
            0: "Relaxing",
            1: "Coffee time",
            2: "Early morning",
            3: "Cleanup",
            4: "Sandwich time",
        },
        "casas": {
            0: "Fill dispenser",
            1: "Hang clothes",
            2: "Move furniture",
            3: "Sit & read",
            4: "Water plants",
            5: "Sweep floor",
            6: "Play checkers",
            7: "Set ingredients",
            8: "Set table",
            9: "Read magazine",
            10: "Pay electric bill",
            11: "Picnic food",
            12: "Get dishes",
            13: "Pack supplies",
            14: "Pack food & deliver",
        },
        # "aras": {
        #     0: "Other",
        #     1: "Going Out",
        #     2: "Preparing Breakfast",
        #     3: "Having Breakfast",
        #     4: "Preparing Lunch",
        #     5: "Having Lunch",
        #     6: "Preparing Dinner",
        #     7: "Having Dinner",
        #     8: "Washing Dishes",
        #     9: "Having Snack",
        #     10: "Sleeping",
        #     11: "Watching TV",
        #     12: "Studying",
        #     13: "Having Shower",
        #     14: "Toileting",
        #     15: "Napping",
        #     16: "Using Internet",
        #     17: "Reading Book",
        #     18: "Laundry",
        #     19: "Shaving",
        #     20: "Brushing Teeth",
        #     21: "Talking on the Phone",
        #     22: "Listening to Music",
        #     23: "Cleaning",
        #     24: "Having Conversation",
        #     25: "Having Guest",
        #     26: "Changing Clothes"
        # },
        "aras": { #[3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 22]
            0: "Preparing Breakfast",
            1: "Having Breakfast",
            2: "Preparing Lunch",
            3: "Preparing Dinner",
            4: "Having Dinner",
            5: "Having Snack",
            6: "Sleeping",
            7: "Watching TV",
            8: "Studying",
            9: "Using Internet",
            10: "Talking on the Phone"
        },
        "openpack": {
            0: "Picking",
            1: "Relocate item label",
            2: "Assemble box",
            3: "Insert items",
            4: "Close box",
            5: "Attach box label",
            6: "Scan label",
            7: "Attach shipping label",
            8: "Put on back table",
            9: "Fill out order",
        }
    }
    return label_mappings.get(dataset_name, {})

def print_kfold_summary(fold_metrics: dict, early_acc_by_step: dict):
    logging.info("\n=== K-Fold Metric Summary ===")
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        logging.info(f"{metric.capitalize():<10}: {mean_val:.4f} ± {std_val:.4f}")

    logging.info("\n=== K-Fold Early Classification Summary ===")
    for step in sorted(early_acc_by_step.keys()):
        acc_list = early_acc_by_step[step]
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        logging.info(f"Step {int(step*100):>3}% → {mean_acc:.4f} ± {std_acc:.4f}")

def plot_two_class_accuracy_by_change(dataset_name, csv_path, best_hm, save_path=None):
    """
    Plots the two classes with the smallest and largest accuracy change between first and last steps.

    Args:
        dataset_name (str): Dataset name to resolve activity labels.
        csv_path (str): Path to CSV file with columns ['step', 'class', 'accuracy'].
        best_hm (float): Best harmonic mean step (percentage).
        save_path (str or None): Path to save the plot. If None, shows the plot.
    """
    df = pd.read_csv(csv_path)
    label_names = get_label_names(dataset_name)

    # Compute mean and std per class and step
    mean_df = df.groupby(['step', 'class'])['accuracy'].mean().reset_index()
    std_df = df.groupby(['step', 'class'])['accuracy'].std().reset_index()

    # Merge for combined plotting
    merged_df = pd.merge(mean_df, std_df, on=['step', 'class'], suffixes=('_mean', '_std'))

    # Identify two classes with the largest and smallest accuracy change over time
    classes = merged_df['class'].unique()
    diffs = {}
    colors = {'min': 'navy', 'max': 'red'}
    for cls in classes:
        class_data = merged_df[merged_df['class'] == cls].sort_values('step')
        first_acc = class_data.iloc[0]['accuracy_mean']
        last_acc = class_data.iloc[-1]['accuracy_mean']
        diffs[cls] = abs(last_acc - first_acc)

    max_diff_class = max(diffs, key=diffs.get)
    min_diff_class = min(diffs, key=diffs.get)

    plt.figure(figsize=(10, 6))

    best_hm_per_class = {}
    for cls in classes:
        class_data = merged_df[merged_df['class'] == cls].sort_values('step')
        steps = class_data['step']
        acc_mean = class_data['accuracy_mean']
        earliness = 1.0 - (steps / 100.0)
        harmonic_mean = 2 * (acc_mean * earliness) / (acc_mean + earliness + 1e-8)
        best_step_idx = np.argmax(harmonic_mean)
        best_step = steps.iloc[best_step_idx]
        best_hm_per_class[cls] = {
            'best_hm': harmonic_mean.iloc[best_step_idx],
            'best_step': best_step,
            'label': label_names.get(cls, f"Class {cls}")
        }

    mean_best_hm = np.mean([info['best_hm'] for info in best_hm_per_class.values()])

    for cls in [min_diff_class, max_diff_class]:
        class_data = merged_df[merged_df['class'] == cls].sort_values('step')
        steps = class_data['step']
        acc_mean = class_data['accuracy_mean']
        acc_std = class_data['accuracy_std']
        earliness = 1.0 - (steps / 100.0)
        harmonic_mean = 2 * (acc_mean * earliness) / (acc_mean + earliness + 1e-8)
        best_step_idx = np.argmax(harmonic_mean)
        best_step = steps.iloc[best_step_idx]

        label = label_names.get(cls, f"Class {cls}")
        color_key = 'min' if cls == min_diff_class else 'max'
        #line_obj, = plt.plot(steps, acc_mean, label=f"{label} (Δ={diffs[cls]:.2f})", color=colors[color_key])
        line_obj, = plt.plot(steps, acc_mean, label=f"{label}", color=colors[color_key])
        plt.fill_between(steps, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color=colors[color_key])
        plt.axvline(x=best_step, linestyle='--', linewidth=2, color=colors[color_key], alpha=0.7)
    # Draw vertical line for best overall HM step
    plt.axvline(x=best_hm, linestyle='-', linewidth=2.5, color='gray', alpha=0.9)

    plt.xlabel("Time Step (%)", fontsize=24)
    plt.ylabel("Accuracy", fontsize=24)
    #plt.legend(title="Activity (Δ Accuracy Difference)", fontsize=22, title_fontsize=22)
    plt.legend(fontsize=24)
    plt.tick_params(axis='both', labelsize=24)
    
    ax = plt.gca()
    ax_right = ax.twinx()
    ax_right.set_yticks([])
    ax_right.set_ylabel("")
    # ax_right.annotate(
    #     "Accuracy Change Over Time Steps",
    #     xy=(1.02, 0.5),
    #     xycoords='axes fraction',
    #     fontsize=24,
    #     va='center',
    #     ha='left',
    #     rotation=270
    # )
    plt.grid(True)


    if save_path:
        two_class_save_path = save_path.replace(".png", "_two_classes.png")
        plt.savefig(two_class_save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return mean_best_hm

# def plot_two_class_accuracy_by_change(merged_df, label_names, save_path=None):
#     """
#     Plots the two classes with the smallest and largest accuracy variation (max-min) across time steps.

#     Args:
#         merged_df (pd.DataFrame): DataFrame containing ['step', 'class', 'accuracy_mean', 'accuracy_std'].
#         label_names (dict): Mapping from class index to label name.
#         save_path (str or None): Path to save the plot. If None, just show.
#     """
#     classes = merged_df['class'].unique()
    
#     # 최대-최소 accuracy 차이 계산
#     diffs = {}
#     for cls in classes:
#         class_data = merged_df[merged_df['class'] == cls]
#         max_acc = class_data['accuracy_mean'].max()
#         min_acc = class_data['accuracy_mean'].min()
#         diffs[cls] = abs(max_acc - min_acc)

#     max_diff_class = max(diffs, key=diffs.get)
#     min_diff_class = min(diffs, key=diffs.get)

#     # Plot
#     plt.figure(figsize=(10, 6))
#     for cls in [min_diff_class, max_diff_class]:
#         class_data = merged_df[merged_df['class'] == cls].sort_values('step')
#         steps = class_data['step']
#         acc_mean = class_data['accuracy_mean']
#         acc_std = class_data['accuracy_std']

#         label = label_names.get(cls, f"Class {cls}")
#         plt.plot(steps, acc_mean, label=f"{label} (Δ={diffs[cls]:.2f})")
#         plt.fill_between(steps, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2)

#     plt.xlabel("Time Step (%)")
#     plt.ylabel("Accuracy")
#     plt.legend(title="Activity (Δ Accuracy)")
#     plt.grid(True)
#     plt.title("Classes with Largest and Smallest Δ (Max-Min) in Accuracy")

#     if save_path:
#         two_class_save_path = save_path.replace(".png", "_two_classes.png")
#         plt.savefig(two_class_save_path, bbox_inches='tight', dpi=300)
#     else:
#         plt.show()

def plot_classwise_early_accuracy_with_std(dataset_name, csv_path, save_path=None):
    """
    Plots class-wise early classification accuracy with optional standard deviation shading.

    Args:
        dataset_name (str): Dataset name to resolve activity labels.
        csv_path (str): Path to CSV file with columns ['step', 'class', 'accuracy'].
        save_path (str or None): If provided, saves the figure to the path.
    """
    df = pd.read_csv(csv_path)
    label_names = get_label_names(dataset_name)

    # Compute mean and std per class and step
    mean_df = df.groupby(['step', 'class'])['accuracy'].mean().reset_index()
    std_df = df.groupby(['step', 'class'])['accuracy'].std().reset_index()

    # Merge for combined plotting
    merged = pd.merge(mean_df, std_df, on=['step', 'class'], suffixes=('_mean', '_std'))

    # Plot class-wise trends
    plt.figure(figsize=(10, 6))
    classes = merged['class'].unique()

    for cls in classes:
        class_data = merged[merged['class'] == cls].sort_values('step')
        steps = class_data['step']
        acc_mean = class_data['accuracy_mean']
        # acc_std = class_data['accuracy_std']  # Enable if shaded region is needed

        label = label_names.get(cls, f"Class {cls}")
        plt.plot(steps, acc_mean, label=label)
        # plt.fill_between(steps, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2)

    plt.xlabel("Time Step (%)")
    plt.ylabel("Accuracy")
    plt.legend(title="Activity")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def save_kfold_summary_to_csv(
    dataset_name: str,
    fold_metrics: dict,
    early_acc_by_step: dict,
    early_acc_by_fold: list,
    early_classwise_acc_by_fold: list,
    output_dir: str = "results"
):
    """
    Save per-fold results, summary statistics, and classwise early accuracy.
    Also generates plots including the best HM step indicator.
    """
    # Prepare output paths
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    per_fold_path = os.path.join(dataset_dir, "per_fold_results.csv")
    summary_path = os.path.join(dataset_dir, "kfold_summary.csv")
    classwise_path = os.path.join(dataset_dir, "classwise_early_accuracy.csv")
    classwise_plot_path = os.path.join(dataset_dir, "classwise_early_plot.png")
    best_hm_path = os.path.join(dataset_dir, "best_hm.txt")

    # 1. Save per-fold metrics (including early accuracy)
    _save_per_fold_results(per_fold_path, fold_metrics, early_acc_by_fold)

    # 2. Save summary statistics and extract best HM step
    best_hm = _save_summary_with_harmonic_mean(summary_path, fold_metrics, early_acc_by_step, best_hm_path)
    mean_best_hm = 0
    
    # 3. Save classwise early accuracy and plot
    if early_classwise_acc_by_fold:
        _save_classwise_early_accuracy(classwise_path, early_classwise_acc_by_fold)

        # Plot mean + std over time per class
        plot_classwise_early_accuracy_with_std(
            dataset_name,
            classwise_path,
            save_path=classwise_plot_path
        )

        # Plot 2 most changed classes and best HM point
        if best_hm is not None:
            mean_best_hm = plot_two_class_accuracy_by_change(
                dataset_name,
                classwise_path,
                best_hm,
                save_path=classwise_plot_path
            )
            

    return best_hm, mean_best_hm

def _save_per_fold_results(per_fold_path, fold_metrics, early_acc_by_fold):
    data = {
        "fold": list(range(len(fold_metrics["accuracy"]))),
        "accuracy": fold_metrics["accuracy"],
        "precision": fold_metrics["precision"],
        "recall": fold_metrics["recall"],
        "f1_score": fold_metrics["f1_score"]
    }
    if early_acc_by_fold:
        step_keys = sorted(early_acc_by_fold[0].keys())
        for step in step_keys:
            step_key = f"early_acc@{int(step * 100)}%"
            data[step_key] = [fold_result.get(step, np.nan) for fold_result in early_acc_by_fold]

    pd.DataFrame(data).to_csv(per_fold_path, index=False)

def _save_summary_with_harmonic_mean(summary_path, fold_metrics, early_acc_by_step, best_hm_path):
    summary = {"metric": [], "mean": [], "std": []}

    # Base metrics
    for metric, values in fold_metrics.items():
        summary["metric"].append(metric)
        summary["mean"].append(np.mean(values))
        summary["std"].append(np.std(values))

    # Early accuracy and harmonic mean
    for step in sorted(early_acc_by_step.keys()):
        step_percent = int(step * 100)
        acc_values = np.array(early_acc_by_step[step])
        earliness_value = 1.0 - step
        hm_values = 2 * (acc_values * earliness_value) / (acc_values + earliness_value + 1e-8)

        summary["metric"].append(f"early_acc@{step_percent}%")
        summary["mean"].append(np.mean(acc_values))
        summary["std"].append(np.std(acc_values))

        summary["metric"].append(f"hm@{step_percent}%")
        summary["mean"].append(np.mean(hm_values))
        summary["std"].append(np.std(hm_values))

    df = pd.DataFrame(summary)
    df.to_csv(summary_path, index=False)

    # Determine best HM step
    hm_rows = df[df["metric"].str.startswith("hm@")]
    if hm_rows.empty:
        return None
    best_row = hm_rows.loc[hm_rows["mean"].idxmax()]
    best_hm = float(best_row["metric"].split("@")[1].replace("%", ""))

    with open(best_hm_path, "w") as f:
        f.write(f"{best_hm:.1f}")

    return best_hm

def _save_classwise_early_accuracy(classwise_path, early_classwise_acc_by_fold):
    records = []
    for fold_idx, fold_classwise in enumerate(early_classwise_acc_by_fold):
        for step, class_acc_dict in fold_classwise.items():
            for cls, acc in class_acc_dict.items():
                records.append({
                    "fold": fold_idx,
                    "step": int(step * 100),
                    "class": cls,
                    "accuracy": acc
                })

    pd.DataFrame(records).to_csv(classwise_path, index=False)

