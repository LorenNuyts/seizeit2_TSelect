import math
import os
from typing import List

import numpy as np

from analysis.utils import extract_values_std_from_results, get_unique_config_names, pretty_print_metrics
from net.DL_config import Config

base_dir = os.path.dirname(os.path.realpath(__file__))

style_map = [ ("*", "solid"), ("square*", "dashed"), ("triangle*", "densely dotted"), ("star", "dashdotted"),
     ("diamond*", "dashdotdotted"),
]

# Example color map — replace with your actual RGB definitions
# These should match the definitions in your document preamble.
color_map = ["tabblue", "taborange","tabgreen", "tabred", "tabpurple",
]

NB_XTICKS = 4

def plot_varying_thresholds_latex(
        configs: List[Config],
        metrics: List[str],
        output_path: str,
        rmsa_filtering: bool = True,
        split_localization: bool = False,
        plots_per_row: int = 3
    ):

    full_to_short_names = get_unique_config_names(configs)

    if split_localization:
        all_lateralizations = ['left', 'right', 'unknown', 'no_seizures', 'bilateral']
    else:
        all_lateralizations = [None]

    fah_epoch_included = "fah_epoch" in metrics
    if not fah_epoch_included:
        metrics.append("fah_epoch")

    for lat in all_lateralizations:

        data, thresholds, lower_bound, upper_bound = extract_values_std_from_results(
            base_dir, configs, full_to_short_names, lat, metrics, rmsa_filtering
        )

        # Count subplots
        effective_metrics = [
            m for m in metrics if not (m == "fah_epoch" and not fah_epoch_included)
        ]
        total_plots = len(effective_metrics)
        rows = math.ceil(total_plots / plots_per_row)

        # Begin groupplot (legend will go ABOVE!)
        header = rf"\begin{{tikzpicture}}"
        header += rf"""

% ================= START GROUPPLOT =================
\begin{{groupplot}}[
    group style={{group size={plots_per_row} by {rows}, horizontal sep=1.5cm, vertical sep=2cm,
                      group name=myplot,}},
    width=\textwidth/{plots_per_row},
    xticklabel style={{font=\small}},
    yticklabel style={{font=\small}},
]
"""

        # ------------- BUILD SUBPLOTS -------------
        body = ""

        for nb_metric, metric in enumerate(effective_metrics):
            parts = []

            max_y_value = 0.0
            min_y_value = 0.0
            for i, (label, values) in enumerate(data[metric].items()):
                avg = values["average"][lower_bound:upper_bound]
                if max(avg) > max_y_value:
                    max_y_value = max(avg)
                if min(avg) < min_y_value:
                    min_y_value = min(avg)
                std = values["std"][lower_bound:upper_bound]
                parts.append(generate_tikz_block(label, thresholds, avg, std, i, add_label=(nb_metric == 0)))

            metric_title = pretty_print_metrics[metric].replace('%', '\%')
            joined = "\n\n".join(parts)
            max_th = max(thresholds)
            min_th = min(thresholds)
            nb_ticks_per_half = math.ceil(NB_XTICKS/2)
            xticks = np.linspace(min_th, 0.5, nb_ticks_per_half, endpoint=False)
            xticks = np.unique(np.append(xticks, np.linspace(0.5, max_th, nb_ticks_per_half)))
            xticks = np.round(xticks, 1)


            body += rf"""
\nextgroupplot[
    title={{\normalsize \textbf{metric_title}}},
    {'xlabel=Decision threshold,' if nb_metric // plots_per_row == rows - 1 else ''} xtick={{{', '.join(map(str, xticks))}}},
]
{joined}
\draw[color=black, dashed, thick] (axis cs:0.5, {-max(2*abs(min_y_value), 2*abs(max_y_value))}) -- (axis cs:0.5, {max(2*abs(min_y_value), 2*abs(max_y_value))});"
"""

        footer = r"""
\end{groupplot}

% ================= BOXED LEGEND PLACED ABOVE FIGURE =================
\path (myplot c1r1.north west|-current bounding box.north)--
      coordinate(legendpos)
      (myplot c2r1.north east|-current bounding box.north);

\matrix[
    matrix of nodes,
    anchor=south,
    draw,
    inner sep=0.3em,
] at ([yshift=1ex]legendpos)
{
"""

        # Python loop (inside your main function)

        all_labels = list(next(iter(data.values())).keys())
        for i, lbl in enumerate(all_labels):
            label_id = lbl.replace(" ", "_").replace("%", "")
            safe_label = lbl.replace("%", "\\%")

            footer += rf"    \ref{{plots:{label_id}}} & {safe_label} & [5pt] \\" + "\n"

        footer += r"""};
        \end{tikzpicture}
        """

        final_tex = header + body + footer

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_tex)

        print(final_tex)
        return


def generate_tikz_block(label, thresholds, averages, stds, idx_label, add_label=False):

    marker, linestyle = style_map[idx_label]
    color = color_map[idx_label]

    coords_main = " ".join(f"({t}, {a})" for t, a in zip(thresholds, averages))
    coords_top = " ".join(f"({t}, {a + s})" for t, a, s in zip(thresholds, averages, stds))
    coords_down = " ".join(f"({t}, {a - s})" for t, a, s in zip(thresholds, averages, stds))

    label_id = label.replace(" ", "_").replace("%", "")
    safe_label = label.replace("%", "\\%")

    return "\n".join([
        rf"\addplot[name path={label_id}_top, draw=none] coordinates {{ {coords_top} }};",
        rf"\addplot[name path={label_id}_down, draw=none] coordinates {{ {coords_down} }};",
        rf"\addplot[{color}, fill opacity=0.2] fill between[of={label_id}_top and {label_id}_down];",
        rf"\addplot[{linestyle}, line width=1.2pt, {color}] coordinates {{ {coords_main} }};",
        rf"\label{{plots:{label_id}}}" if add_label else "",
    ])
