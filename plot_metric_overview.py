import os
import numpy as np
import pandas as pd
import configparser
from metric_visualisation import (get_metrics, create_embedding, plot_thematic_metric_embedding_heatmap,
                                  plot_metric_embedding_heatmap, plot_metric_criteria_occurrence_distribution,
                                  plot_aggregation_metric_distribution, plot_measurement_scale_distribution,
                                  plot_scenario_1, plot_scenario_2, plot_scenario_3, plot_overlapping_metrics)


def main():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'paths.ini'))

    out_dir = os.path.abspath(config['paths']['output_directory'])
    endpoint = config['Graph database']['sparql_endpoint']

    metrics = get_metrics(endpoint)
    colors = ["#d5eeb3", "#2599c2", "#0b5313", "#a7dcf9", "#42952e", "#90ea66", "#22577a", "#57ecc0"]
    metrics_with_criteria = metrics.loc[~metrics['criteria_type'].isnull(),:]

    thematic_embedding = create_embedding(metrics, 'thematic_metric')
    criteria_embedding = create_embedding(metrics_with_criteria, 'metric_type')

    # descriptive analysis
    fig, _ = plot_thematic_metric_embedding_heatmap(thematic_embedding, 12, 5)
    fig.savefig(os.path.join(out_dir, 'thematic_metric_embedding_heatmap.png'))

    fig, _ = plot_metric_embedding_heatmap(criteria_embedding.loc[:, criteria_embedding.sum(axis=0) > 1], 12, 16)
    fig.savefig(os.path.join(out_dir, 'criteria_embedding_heatmap.png'))

    fig, _ = plot_metric_criteria_occurrence_distribution(criteria_embedding, colors, 6, 6)
    fig.savefig(os.path.join(out_dir, 'criteria_occurrence_distribution.png'))

    fig, _ = plot_aggregation_metric_distribution(metrics, colors, 6, 6)
    fig.savefig(os.path.join(out_dir, 'aggregation_metric_distribution.png'))

    fig, _ = plot_measurement_scale_distribution(metrics, colors, 3, 6)
    fig.savefig(os.path.join(out_dir, 'measurement_scale_distribution.png'))

    # Scenarios
    labels = metrics['metric_type'].value_counts()[metrics['metric_type'].value_counts() > 1].index
    twice_metrics = metrics[metrics['metric_type'].isin(list(labels))]
    features = pd.Series(np.zeros(len(twice_metrics['representation_feature'].dropna().unique())), index=twice_metrics['representation_feature'].dropna().unique())
    criteria = pd.Series(np.zeros(len(twice_metrics['criteria_type'].dropna().unique())), index=twice_metrics['criteria_type'].dropna().unique())
    thematic_metrics = pd.Series(np.zeros(len(twice_metrics['thematic_metric'].dropna().unique())), index = twice_metrics['thematic_metric'].dropna().unique())

    # scenario 1
    s1_metrics, fig, _ = plot_scenario_1(twice_metrics, features, thematic_metrics, criteria, labels, colors, 15, 5)
    fig.savefig(os.path.join(out_dir, 'scenario_1.png'))

    # scenario 2
    s2_metrics, fig, _ = plot_scenario_2(twice_metrics, features, thematic_metrics, criteria, labels, colors, 15, 5)
    fig.savefig(os.path.join(out_dir, 'scenario_2.png'))

    # scenario 3
    s3_metrics, fig, _ = plot_scenario_3(twice_metrics, features, thematic_metrics, criteria, labels, colors, 15, 5)
    fig.savefig(os.path.join(out_dir, 'scenario_3.png'))

    # scenario overlap
    fig, _ = plot_overlapping_metrics(s1_metrics, s2_metrics, s3_metrics, colors[:3], 6, 4)
    fig.savefig(os.path.join(out_dir, 'overlapping_metrics.png'))

if __name__ == "__main__":
    main()
