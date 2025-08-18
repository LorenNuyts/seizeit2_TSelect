import itertools
import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from net.DL_config import Config
from utility.paths import get_path_config, get_path_results
from utility.stats import Results


def count_selected_channels_across_folds(base_dir, configs: List[Config], output_path: str = None):
    for config in configs:
        config_path = os.path.join(base_dir, "..", "..", get_path_config(config, config.get_name()))
        if os.path.exists(config_path):
            config.load_config(config_path, config.get_name())
        else:
            print(f"Config not found for {config.get_name()}")
            continue
        nb_folds = config.nb_folds
        channels = config.selected_channels
        channel_count = {}
        for fold in channels:
            for ch in channels[fold]:
                if ch not in channel_count:
                    channel_count[ch] = 1
                else:
                    channel_count[ch] += 1

        sorted_channels = sorted(channel_count.items(), key=lambda x: x[1], reverse=True)
        print(f"Number of folds a channel is chosen: {sorted_channels}")
        # make a hbar plot with channels on y-axis and counts on x-axis + sort them
        channels, counts = zip(*sorted_channels)

        # Create horizontal bar plot
        plt.figure(figsize=(10, 6))
        plt.barh(channels, counts, color='skyblue')
        plt.xlabel("Number of folds the channel is selected out of the {} folds".format(nb_folds))
        plt.title("Channel Counts")
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top

        if output_path:
            output_file = os.path.join(output_path, "channel_counts", f"{config.get_name()}_channel_counts.png")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file)
        else:
            plt.show()

        plt.close()

def mine_frequent_channels(base_dir, configs: List[Config], output_path: str = None, min_support: float = 0.5):
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    for config in configs:
        config_path = os.path.join(base_dir, "..", "..", get_path_config(config, config.get_name()))
        if os.path.exists(config_path):
            config.load_config(config_path, config.get_name())
        else:
            print(f"Config not found for {config.get_name()}")
            continue
        channels = config.selected_channels
        transactions = []
        for fold in channels:
            transactions.append(list(channels[fold]))

        # Convert transactions to a one-hot encoded DataFrame
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Apply Apriori algorithm
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_support)

        print(f"Frequent itemsets for {config.get_name()}:")
        print(frequent_itemsets)
        print(f"Association rules for {config.get_name()}:")
        print(rules)

        if output_path:
            output_file = os.path.join(output_path, "frequent_channels", f"{config.get_name()}_frequent_channels.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Save the frequent itemsets and rules to same CSV file
            frequent_itemsets.to_csv(output_file, index=False)
            with open(output_file, 'a') as f:
                f.write('\n')
            rules.to_csv(output_file, index=False, mode='a')

def find_interchangeable_channels(base_dir, configs: List[Config], output_path:str):
    for i, config in enumerate(configs):
        config_path = os.path.join(base_dir, "..", "..", get_path_config(config, config.get_name()))
        if os.path.exists(config_path):
            config.load_config(config_path, config.get_name())
        else:
            print(f"Config not found for {config.get_name()}")
            continue
        # results = []
        # results_path = os.path.join(base_dir, "..", "..", get_path_results(config, config.get_name()))
        # results.append(Results(config))
        # if os.path.exists(results_path):
        #     results[i].load_results(results_path)
        # else:
        #     print(f"Results not found for {config.get_name()}")
        #     continue
        #
        # result = results[i]
        included_channels = sorted(config.included_channels)
        # TODO: possibly change these two lines
        selected_channels = [config.selected_channels[fold_i] for fold_i in range(config.nb_folds)]
        clusters = [config.channel_selector[fold_i].clusters for fold_i in range(config.nb_folds)]
        # Count how many times two channels appears in the same cluster
        co_occurrence_matrix = np.zeros((len(included_channels), len(included_channels)))
        for clustering_i in clusters:
            for cluster in clustering_i:
                for ch1, ch2 in itertools.combinations(cluster, 2):
                    co_occurrence_matrix[ch1, ch2] += 1
                    co_occurrence_matrix[ch2, ch1] += 1

                for ch in cluster:
                    co_occurrence_matrix[ch, ch] += 1

        # Normalize the co-occurrence matrix by the number of folds
        co_occurrence_matrix /= config.nb_folds

        # Average the evaluation metric for channel selection across the folds to get an indication of how relevant
        # the channel is
        average_relevance_channels = {ch: 0 for ch in range(len(included_channels))}
        for fold_i in range(config.nb_folds):
            evaluation_metrics = config.channel_selector[fold_i].evaluation_metric_per_channel
            evaluation_metrics.update(config.channel_selector[fold_i].removed_series_too_low_metric)
            for ch, metric in evaluation_metrics.items():
                average_relevance_channels[ch] += metric
        # Average the relevance across folds
        average_relevance_channels = {ch: metric / config.nb_folds for ch, metric in average_relevance_channels.items()}

        # Save selected channels, co-occurrence matrix and average_relevance to a CSV file
        output_file = os.path.join(output_path, f"{config.get_name()}_interchangeable_channels.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with pd.ExcelWriter(output_file) as writer:
            selected_channels_df = pd.DataFrame(selected_channels, index=config.selected_channels.keys(), columns=['Selected Channels'])
            selected_channels_df.to_excel(writer, sheet_name='Selected Channels')
            co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=included_channels, columns=included_channels)
            co_occurrence_df.to_excel(writer, sheet_name='Co-occurrence Matrix')
            average_relevance_df = pd.DataFrame(list(average_relevance_channels.values()),
                                                columns=['Average Relevance'],
                                                index=included_channels)
            average_relevance_df.to_excel(writer, sheet_name='Average Relevance')
            print(f"Interchangeable channels for {config.get_name()} saved to {output_file}")

        # Combine the count and averaged evaluation metric to find the channels that are equally relevant, but redundant


def map_clusters_between_folds(clusters, correlation_threshold=0.7):
    result = np.ndarray(shape=(len(clusters), len(clusters)))
    for i in range(len(clusters)):
        for j in range(i, len(clusters)):
            # If same fold, map all clusters to themselves
            if i == j:
                result[i,j] = {c:c for c in range(len(clusters[i]))}
                continue
            # If different folds, map clusters based on correlation




