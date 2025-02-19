import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

import pandas as pd
import numpy as np
import math
import itertools
import re
import json
import importlib

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

def print_title(title, line_length = 60, symbol = '-'):
    separator = symbol * ((line_length - len(title) - 2) // 2)
    print(f"{separator} {title} {separator}")

def format_columns(df, datetime_columns=[], int64_columns=[], float64_columns=[], str_columns=[]):
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    for col in int64_columns:
        if col in df.columns:
            df[col] = df[col].astype('Int64')  

    for col in float64_columns:
        if col in df.columns:
            df[col] = df[col].astype('float64')

    for col in str_columns:
        if col in df.columns:
            df[col] = df[col].astype('str')
    
    return df

def unbalance_dataset(selected_df, variant_column, imbalance_factor):
    counts = selected_df[variant_column].value_counts()
    max_count = counts.max()
    max_variants = counts[counts == max_count].index.tolist()
    desired_count = int(max_count / imbalance_factor)
    frames = []
    for variant, _ in counts.items():
        df_variant = selected_df[selected_df[variant_column] == variant]
        if variant in max_variants:
            frames.append(df_variant)
        else:
            if len(df_variant) < desired_count:
                df_variant = df_variant.sample(n=desired_count, random_state=42, replace=True)
            else:
                df_variant = df_variant.sample(n=desired_count, random_state=42, replace=False)
            frames.append(df_variant)
    return pd.concat(frames).reset_index(drop=True)

def extract_p_value(json_data):
    first_key = next(iter(json_data))
    p_value = json_data[first_key].get("p_value")
    return p_value

def plot_distributions(selected_df, variant_column, metric_column, alpha=0.05):
    variants = selected_df[variant_column].unique()
    z = norm.ppf(1 - alpha/2)  # Critical value based on alpha
    
    if len(variants) == 2:
        plt.figure(figsize=(16, 4))
        for variant in variants:
            subset = selected_df[selected_df[variant_column] == variant]
            mean_val = subset[metric_column].mean()
            std_val = subset[metric_column].std()
            n = len(subset)
            se = std_val / math.sqrt(n)
            lower_ci = mean_val - z * se
            upper_ci = mean_val + z * se
            x_min = lower_ci - 0.2 * (upper_ci - lower_ci)
            x_max = upper_ci + 0.2 * (upper_ci - lower_ci)
            x_values = np.linspace(x_min, x_max, 200)
            pdf = norm.pdf(x_values, mean_val, se)
            plt.plot(x_values, pdf, lw=2, label=variant)
            plt.axvline(mean_val, color='green', linestyle='-',alpha=0.7)
            plt.axvline(lower_ci, color='red', linestyle='--', alpha=0.5)
            plt.axvline(upper_ci, color='red', linestyle='--', alpha=0.5)
            plt.fill_between(x_values, pdf, where=(x_values >= lower_ci) & (x_values <= upper_ci),
                             color='red', alpha=0.2)
            pdf_mean = norm.pdf(mean_val, mean_val, se)
            plt.text(mean_val, pdf_mean * 1.05, f"{mean_val:.2f}",
                     ha='center', va='bottom', color='green', fontsize=10)
            pdf_lower = norm.pdf(lower_ci, mean_val, se)
            pdf_upper = norm.pdf(upper_ci, mean_val, se)
            plt.text(lower_ci, pdf_lower * 1.05, f"Lower CI: {lower_ci:.2f}",
                     ha='center', va='bottom', color='red', fontsize=10)
            plt.text(upper_ci, pdf_upper * 1.05, f"Upper CI: {upper_ci:.2f}",
                     ha='center', va='bottom', color='red', fontsize=10)
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.legend(title="Variant", fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.show()
        
    else:
        pairs = list(itertools.combinations(variants, 2))
        n_pairs = len(pairs)
        fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 4 * n_pairs))
        if n_pairs == 1:
            axes = [axes]
        for idx, pair in enumerate(pairs):
            ax = axes[idx]
            for variant in pair:
                subset = selected_df[selected_df[variant_column] == variant]
                mean_val = subset[metric_column].mean()
                std_val = subset[metric_column].std()
                n = len(subset)
                se = std_val / math.sqrt(n)
                lower_ci = mean_val - z * se
                upper_ci = mean_val + z * se
                x_min = lower_ci - 0.2 * (upper_ci - lower_ci)
                x_max = upper_ci + 0.2 * (upper_ci - lower_ci)
                x_values = np.linspace(x_min, x_max, 200)
                pdf = norm.pdf(x_values, mean_val, se)
                ax.plot(x_values, pdf, lw=2, label=variant)
                ax.axvline(mean_val, color='green', linestyle='-', alpha=0.7)
                ax.axvline(lower_ci, color='red', linestyle='--', alpha=0.5)
                ax.axvline(upper_ci, color='red', linestyle='--', alpha=0.5)
                ax.fill_between(x_values, pdf, where=(x_values >= lower_ci) & (x_values <= upper_ci),
                                color='red', alpha=0.2)
                pdf_mean = norm.pdf(mean_val, mean_val, se)
                ax.text(mean_val, pdf_mean * 1.05, f"{mean_val:.2f}",
                        ha='center', va='bottom', color='green', fontsize=10)
                pdf_lower = norm.pdf(lower_ci, mean_val, se)
                pdf_upper = norm.pdf(upper_ci, mean_val, se)
                ax.text(lower_ci, pdf_lower * 1.05, f"Lower CI: {lower_ci:.2f}",
                        ha='center', va='bottom', color='red', fontsize=10)
                ax.text(upper_ci, pdf_upper * 1.05, f"Upper CI: {upper_ci:.2f}",
                        ha='center', va='bottom', color='red', fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(title="Variant", fontsize=10, title_fontsize=12)
        plt.tight_layout()
        plt.show()

def add_segment_column(selected_df, num_segments=3):
    unique_segments = [f"segment_{i+1}" for i in range(num_segments)]
    
    # Generating random segments
    np.random.seed(42)  # For reproducibility
    selected_df['segment'] = np.random.choice(unique_segments, size=len(selected_df))
    
    return selected_df