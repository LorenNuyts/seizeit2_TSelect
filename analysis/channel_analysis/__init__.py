import os
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from net.DL_config import Config
from utility.paths import get_path_config

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