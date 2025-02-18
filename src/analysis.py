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

import pingouin as pg
import scikit_posthocs as sp

from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, chi2_contingency, fisher_exact, norm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

def remove_outliers(df, metric_column, factor=1.5, threshold=0.1):
    q1 = df[metric_column].quantile(0.2)
    q3 = df[metric_column].quantile(0.8)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    print("Low outlier limit:", lower_bound)
    print("Upper outlier limit:", upper_bound)

    initial_count = df.shape[0]
    df_filtered = df.loc[(df[metric_column] >= lower_bound) & (df[metric_column] <= upper_bound)]
    final_count = df_filtered.shape[0]
    removed_percentage = (initial_count - final_count) / initial_count

    print(f"Filtered {initial_count - final_count} rows ({removed_percentage:.2%}) from {initial_count} to {final_count}")

    is_strong_outlier_effect = removed_percentage > threshold
    print(f"is_strong_outlier_effect: {is_strong_outlier_effect}")

    return df_filtered, is_strong_outlier_effect

def remove_axes_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

def calculate_distribution(selected_df, variant_column, metric_column):
    results = {}
    variants = selected_df[variant_column].unique()
    for variant in variants:
        metric_data = selected_df.loc[selected_df[variant_column] == variant, metric_column]
        if len(metric_data) < 5000:
            stat, p_value = stats.shapiro(metric_data)
            test_name = 'shapiro'
        else:
            stat, p_value = stats.normaltest(metric_data)
            test_name = 'normaltest'
        results[variant] = {'test': test_name, 'stat': stat, 'p_value': p_value}
        print(f"variant {variant}: {test_name} statistic = {stat:.4f}, p-value = {p_value:.4f}")
    return results

def plot_qq(selected_df, variant_column, metric_column):
    variants = selected_df[variant_column].unique()
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        metric_data = selected_df.loc[selected_df[variant_column] == variant, metric_column]
        stats.probplot(metric_data, dist="norm", plot=ax)
        ax.set_title(f'qq plot - {variant}', fontsize=12)
        remove_axes_frame(ax)
    plt.tight_layout()
    plt.show()

def plot_histogram_kde(selected_df, variant_column, metric_column):
    variants = selected_df[variant_column].unique()
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        metric_data = selected_df.loc[selected_df[variant_column] == variant, metric_column]
        sns.histplot(
            metric_data,
            kde=True,
            ax=ax,
            color='skyblue',
            stat='density',
            edgecolor=None,
            alpha=0.7
        )
        ax.set_title(f'histogram - {variant}', fontsize=12)
        ax.set_xlabel(metric_column)
        ax.set_ylabel('density')
        ax.grid(True, linestyle='--', alpha=0.6)
        remove_axes_frame(ax)
    plt.tight_layout()
    plt.show()

def plot_violin(selected_df, variant_column, metric_column):
    variants = selected_df[variant_column].unique()
    n = len(variants)
    fig, ax = plt.subplots(figsize=(6 * n, 4))
    sns.violinplot(
        x=variant_column,
        y=metric_column,
        data=selected_df,
        hue=variant_column,
        palette='pastel',
        inner='quartile',
        dodge=False,
        ax=ax
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title('violin plot - metric distribution', fontsize=12)
    ax.set_xlabel('variant')
    ax.set_ylabel(metric_column)
    ax.grid(True, linestyle='--', alpha=0.6)
    remove_axes_frame(ax)
    plt.tight_layout()
    plt.show()

def plot_combined_kde(selected_df, variant_column, metric_column):
    variants = selected_df[variant_column].unique()
    n = len(variants)
    fig, ax = plt.subplots(figsize=(6 * n, 4))
    for variant in variants:
        metric_data = selected_df.loc[selected_df[variant_column] == variant, metric_column]
        sns.kdeplot(
            metric_data,
            fill=True,
            label=variant,
            alpha=0.6,
            ax=ax
        )
    ax.set_title('combined kde - metric distribution', fontsize=12)
    ax.set_xlabel(metric_column)
    ax.set_ylabel('density')
    ax.legend(title='variant')
    ax.grid(True, linestyle='--', alpha=0.6)
    remove_axes_frame(ax)
    plt.tight_layout()
    plt.show()

def set_normal_distribution_flag(distribution_results, alpha=0.05):
    # Set flag for normal distribution based on p_value > alpha
    for variant, result in distribution_results.items():
        is_normal_distribution = result['p_value'] > alpha
    return is_normal_distribution

def calculate_variance_analysis(selected_df, variant_column, metric_column, alpha=0.05):
    # Levene's test for homogeneity of variances
    variants = selected_df[variant_column].unique()
    groups = [selected_df.loc[selected_df[variant_column] == variant, metric_column] for variant in variants]
    stat, p_value = stats.levene(*groups, center='median')
    is_equal_variance = p_value > alpha
    print(f"Levene test statistic = {stat:.4f}, p_value = {p_value:.4f}")
    print(f"Equal variance assumption: {is_equal_variance}")
    return {'test': 'levene', 'stat': stat, 'p_value': p_value, 'is_equal_variance': is_equal_variance}

def set_equal_variance_flag(variance_results, alpha=0.05):
    # Set flag for equal variance based on p_value > alpha
    is_equal_variance = variance_results['p_value'] > alpha
    return is_equal_variance