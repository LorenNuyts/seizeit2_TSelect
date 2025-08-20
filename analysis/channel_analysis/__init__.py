import itertools
import os
from collections import defaultdict
from typing import List, Set, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

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

def find_interchangeable_channels(base_dir, configs: List[Config], output_path:str, minimal_support: float = 0.5):
    assert 0 <= minimal_support <= 1
    for i, config in enumerate(configs):
        config_path = os.path.join(base_dir, "..", "..", get_path_config(config, config.get_name()))
        if os.path.exists(config_path):
            config.load_config(config_path, config.get_name())
        else:
            print(f"Config not found for {config.get_name()}")
            continue

        config.nb_folds = len(config.folds)
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
        nb_folds_channel_selected = np.zeros((len(included_channels), len(included_channels)), dtype=int)
        for clustering_i in clusters:
            for cluster in clustering_i:
                for ch1, ch2 in itertools.combinations(cluster, 2):
                    co_occurrence_matrix[ch1, ch2] += 1
                    co_occurrence_matrix[ch2, ch1] += 1

                for ch in cluster:
                    co_occurrence_matrix[ch, ch] += 1
                    nb_folds_channel_selected[ch, ch] += 1

            channels_in_all_clusters = list(itertools.chain.from_iterable(clustering_i))
            for ch1, ch2 in itertools.combinations(channels_in_all_clusters, 2):
                nb_folds_channel_selected[ch1, ch2] += 1
                nb_folds_channel_selected[ch2, ch1] += 1

        # Normalize the co-occurrence matrix by the  number of folds each channel got past the irrelevant selector
        for m in range(co_occurrence_matrix.shape[0]):
            for n in range(m, co_occurrence_matrix.shape[1]):
                co_occurrence_matrix[m, n] /= nb_folds_channel_selected[m, n]
                if m != n:
                    co_occurrence_matrix[n, m] /= nb_folds_channel_selected[m, n]
                    assert co_occurrence_matrix[m, n] == co_occurrence_matrix[n, m] or \
                    (np.isnan(co_occurrence_matrix[m, n]) and np.isnan(co_occurrence_matrix[n, m]))


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

        # Select all channels that have a co-occurrence above a certain threshold
        processed_co_occurrence = process_co_occurrence_matrix(co_occurrence_matrix, minimal_support)

        # Add relevance to results
        def co_occurrence2str(chs: dict) -> Dict[str, dict | str]:
            result = {}
            for key, value in chs.items():
                chs_relevance = {included_channels[c]: average_relevance_channels[c] for c in key}
                sorted_chs = [f"{k} ({'%.2f' % v})" for k, v in sorted(chs_relevance.items(), key=lambda item: item[1], reverse=True)]
                key_str = ', '.join(sorted_chs)

                if value == []:
                    value_str = ""
                else:
                    value_str = co_occurrence2str(value)

                result[key_str] = value_str
            return result

        processed_co_occurrence_with_relevance = co_occurrence2str(processed_co_occurrence)


        # for key, value in processed_co_occurrence.items():
        #
        # for c in processed_co_occurrence:
        #     c_relevance = {included_channels[ch]: average_relevance_channels[ch] for ch in c}
        #     sorted_c = [f"{k}: {'%.2f' % v}" for k, v in sorted(c_relevance.items(), key=lambda item: item[1], reverse=True)]
        #     processed_co_occurrence_with_relevance.append(sorted_c)

        # Save selected channels, co-occurrence matrix and average_relevance to a CSV file
        output_file = os.path.join(output_path, f"{config.get_name()}_interchangeable_channels.xlsx")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            workbook = writer.book
            sheet = workbook.create_sheet("Interchangeable Channels")
            writer.sheets["Interchangeable Channels"] = sheet

            # Title font and alignment
            title_font = Font(bold=True, size=12)
            alignment = Alignment(horizontal="center")

            # Write title and selected channels
            sheet.cell(row=1, column=1, value="Selected Channels").font = title_font
            sheet.cell(row=1, column=1).alignment = alignment

            selected_channels_df = pd.DataFrame(
                selected_channels,
                index=[f"Fold {i}" for i in config.selected_channels.keys()]
            )
            selected_channels_df.to_excel(writer, sheet_name="Interchangeable Channels", startrow=2)

            # Adjust column widths for selected channels
            for col_num in range(1, len(selected_channels_df.columns) + 1):
                col_letter = get_column_letter(col_num)
                sheet.column_dimensions[col_letter].width = 15

            # Write title and processed co-occurrence
            start_row = 2 + len(selected_channels_df) + 4  # Add extra whitespace
            sheet.cell(row=start_row, column=1, value="Processed Co-occurrence with Relevance").font = title_font
            sheet.cell(row=start_row, column=1).alignment = alignment

            start_row += 1
            # Write each list in processed_co_occurrence_with_relevance as a row
            def write_co_occurrence_to_sheet(co_occurrence_dict, sheet, start_row, start_column):
                row_nb = start_row
                for key, value in co_occurrence_dict.items():
                    splitted_key = key.split(", ")
                    for j, v in enumerate(splitted_key, start=start_column):
                        sheet.cell(row=row_nb, column=j, value=v)
                    row_nb += 1
                    if isinstance(value, dict):
                        # If value is a dict, write it recursively
                        extra_rows = write_co_occurrence_to_sheet(value, sheet, row_nb, start_column+1)
                        row_nb += extra_rows
                return row_nb - start_row
            start_row += write_co_occurrence_to_sheet(processed_co_occurrence_with_relevance, sheet, start_row, 1)

            # Write title and co-occurrence matrix
            # start_row += len(processed_co_occurrence_with_relevance) + 4  # Add extra whitespace
            start_row += 4  # Add extra whitespace
            sheet.cell(row=start_row, column=1, value="Co-occurrence Matrix").font = title_font
            sheet.cell(row=start_row, column=1).alignment = alignment

            co_occurrence_df = pd.DataFrame(
                co_occurrence_matrix,
                index=included_channels,
                columns=included_channels
            )
            co_occurrence_df.to_excel(writer, sheet_name="Interchangeable Channels", startrow=start_row + 1)

            # Write title and average relevance
            start_row += len(co_occurrence_df) + 4  # Add extra whitespace
            sheet.cell(row=start_row, column=1, value="Average Relevance").font = title_font
            sheet.cell(row=start_row, column=1).alignment = alignment

            average_relevance_df = pd.DataFrame(
                list(average_relevance_channels.values()),
                columns=["Average Relevance"],
                index=included_channels
            )
            average_relevance_df.to_excel(writer, sheet_name="Interchangeable Channels", startrow=start_row + 1)

            # Adjust column widths for all dataframes
            for col_num in range(1, average_relevance_df.shape[1] + 1):
                col_letter = get_column_letter(col_num)
                sheet.column_dimensions[col_letter].width = 20

        print(f"Interchangeable channels for {config.get_name()} saved to {output_file}")
        # Combine the count and averaged evaluation metric to find the channels that are equally relevant, but redundant


def process_co_occurrence_matrix(co_occurrence: np.ndarray, minimal_support=0.5):
    assert 0 <= minimal_support <= 1

    groups: List[Set] = []

    for i in range(co_occurrence.shape[0]):
        # Find all indices j where co_occurrence[i, j] >= minimal_support
        # and j >= i to avoid duplicates
        group = {i}
        for j in range(i + 1, co_occurrence.shape[1]):
            if co_occurrence[i, j] >= minimal_support:
                group.add(j)

        to_remove = set()
        for m, n in itertools.combinations(group, 2):
            if co_occurrence[m, n] < minimal_support:
                to_remove.add(n)
        group.difference_update(to_remove)

        if not all(np.isnan(co_occurrence[i])):
            # Check if the group is already in groups
            if not any(group.issubset(existing_group) for existing_group in groups):
                # If not, add it to groups
                groups.append(group)

    # index_structure = defaultdict(list)
    # for i, group in enumerate(groups):
    #     for elem in group:
    #         index_structure[elem].append(i)

    def list_to_tree(sets: List[Set]):
        result = dict()
        sets = sorted(sets, key=len)
        already_handled = set()
        for s_i, s in enumerate(sets):
            if s_i in already_handled:
                continue
            common = [sets[k] for k in range(s_i + 1, len(sets)) if len(s.intersection(sets[k])) > 0]
            if not common:
                result[frozenset(s)] = []
            else:
                intersection = set.intersection(*common, s)
                if not intersection:
                    result[frozenset(s)] = []
                    continue
                differences = [other.difference(intersection) for other in [s, *common]]
                to_remove_differences = set()
                for diff_i, diff in enumerate(differences):
                    for elem in diff:
                        others = set.union(*[d for d_i, d in enumerate(differences) if d_i != diff_i])
                        values = [co_occurrence[elem, o] for o in others if (co_occurrence[elem, o] < minimal_support or
                                  np.isnan(co_occurrence[elem, o]))]
                        if all(np.isnan(values)):
                            intersection.add(elem)
                            to_remove_differences.add(elem)

                differences = [{d for d in diff if d not in to_remove_differences} for diff in differences]
                differences = [d for d in differences if len(d) > 0]
                if not differences:
                    result[frozenset(intersection)] = []
                else:
                    result[frozenset(intersection)] = list_to_tree(differences)
                already_handled.update({sets.index(c) for c in common})

        return result

    return list_to_tree(groups)

    # def add_to_partial_result(to_add):
    #     added_at_least_once = False
    #     maximal_possibilities = []
    #     results_of = [for g in groups if add_to_result_of in g]
    #     for result_i in index_structure[add_to_result_of]:
    #         maximal_possibility = groups[result_i].copy()
    #         can_add = True
    #         for elem in groups[result_i]:
    #             if elem == to_add or elem == add_to_result_of:
    #                 continue
    #             # if np.isnan(co_occurrence[elem, to_add]) or co_occurrence[elem, to_add] < minimal_support:
    #             if co_occurrence[elem, to_add] < minimal_support:
    #                 can_add = False
    #                 maximal_possibility.remove(elem)
    #         if can_add:
    #             groups[result_i].add(to_add)
    #             added_at_least_once = True
    #         else:
    #             maximal_possibilities.append(maximal_possibility)
    #     return added_at_least_once, maximal_possibilities
    #
    # groups: List[Set] = []
    #
    # for i in range(co_occurrence.shape[0]):
    #     for j in range(i, co_occurrence.shape[1]):
    #         if co_occurrence[i, j] >= minimal_support:
    #             added_i, possibilities_i = add_to_partial_result(i, j)
    #             added_j, possibilities_j = add_to_partial_result(j, i)
    #             if not added_i and not added_j:
    #                 if len(possibilities_i) == 0 and len(possibilities_j) == 0:
    #                     groups.append({i, j})
    #                 else:
    #                     for p in possibilities_i:
    #                         p.add(i)
    #                         if p not in groups:
    #                             groups.append(p)
    #                     for p in possibilities_j:
    #                         p.add(j)
    #                         if p not in groups:
    #                             groups.append(p)
    #
    #
    #
    # # groups = retain_maximal_sets(groups) # TODO adapt indexing structure!
    # # result = []
    # # for i, indices in index_structure.items():
    # #     if len(indices) > 1:
    # #         results_i = [groups[idx] for idx in indices]
    # #         tree = sets_to_tree(results_i)
    # #     else:
    # #         result.append(groups[indices.pop()])
    #
    # return groups

# def retain_maximal_sets(sets):
#     # Sort sets by length in descending order
#     sorted_sets = sorted(sets, key=len, reverse=True)
#     result = []
#
#     for s in sorted_sets:
#         # Add the set to the result only if it is not a subset of any set already in the result
#         if not any(s < other for other in result):
#             result.append(s)
#
#     return result
#
# def sets_to_tree(sets_):
#     from collections import defaultdict
#
#     def build_tree(sets):
#         if not sets:
#             return None
#
#         # Convert sets to sorted lists for consistent processing
#         sorted_lists = [sorted(s) for s in sets]
#
#         # Find the common elements
#         common_elements = set.intersection(*sets) if sets else set()
#
#         # Remove the common elements from all sets
#         children = defaultdict(list)
#         for s in sets:
#             remaining = s - common_elements
#             if remaining:
#                 children[frozenset(remaining)].append(remaining)
#
#         # Recursively build the tree for each child
#         return {
#             "value": list(common_elements),
#             "children": [build_tree([set(c) for c in child]) for child in children.values()]
#         }
#
#     return build_tree(sets_)

# def co_occurrence_matrix2tree(co_occur)





