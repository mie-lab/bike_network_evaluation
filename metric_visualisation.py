import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib_venn import venn3


def get_metrics(endpoint):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery("""PREFIX nemo:<http://www.ebikecityevaluationtool.com/ontology/nemo#>
    PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>
    PREFIX geo: <http://www.opengis.net/ont/geosparql#>
    SELECT ?metric_type ?method ?thematic_metric ?criteria_type ?representation_feature ?measurement_scale
    WHERE {GRAPH <http://www.ebikecityevaluationtool.com/ontology/nemo/metrics/> { 
    ?metric rdf:type ?metric_type.
    ?metric nemo:usedIn ?method.
    FILTER REGEX(STR(?metric_type),'nemo')
    ?metric_type rdfs:subClassOf ?thematic_metric .
    OPTIONAL{?metric nemo:mapsToFeature/rdf:type ?representation_feature.}
    OPTIONAL{ ?metric nemo:measures ?criteria.
    ?criteria rdf:type ?criteria_type.}
    OPTIONAL {?metric nemo:hasMeasurementScale ?measurement_scale.}}}""")
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    qr = pd.DataFrame(results['results']['bindings'])
    qr = qr.applymap(lambda cell: cell if pd.isnull(cell) else cell['value'])
    prefix = 'http://www.ebikecityevaluationtool.com/ontology/nemo#'

    qr['metric_type'] = qr['metric_type'].str.replace(prefix, '')
    qr['thematic_metric'] = qr['thematic_metric'].str.replace(prefix, '')
    qr['criteria_type'] = qr['criteria_type'].str.replace(prefix, '')
    qr['representation_feature'] = qr['representation_feature'].str.replace(prefix, '')
    qr['measurement_scale'] = qr['measurement_scale'].str.replace(prefix, '')
    qr = qr[qr['representation_feature'] != 'RepresentationFeature']
    qr = qr.sort_values(by='thematic_metric')

    return qr


def create_embedding(metrics, column):

    """
    Args:
        metrics: DataFrame containing the metrics and their properties.
        column: column that contains criteria type.
    Returns: embedding that allows to compare criteria based on associated metric characteristics.
    """

    metric_counts = metrics.groupby(by=['criteria_type', column])['method'].count()
    criteria_embedding = pd.DataFrame(index=metrics['criteria_type'].dropna().unique(), 
                                      columns=metrics[column].unique(), dtype=float)

    for i in metrics['criteria_type'].dropna().unique():
        current_counts = metric_counts.loc[i]
        criteria_embedding.loc[i, current_counts.index] = current_counts.to_numpy()
    criteria_embedding = criteria_embedding.fillna(value=0)

    return criteria_embedding


def plot_thematic_metric_embedding_heatmap(embedding_orig, x, y):
    """
    The plot displays the distribution of thematic metric type counts across existing criteria in the dataset.
    Args:
        embedding_orig: original embedding DataFrame.
        x: width of the plot.
        y: height of the plot.
    """

    embedding = embedding_orig.copy()
    row_sums = embedding.sum(axis=0).sort_values(ascending=True)
    col_sums = embedding.sum(axis=1).sort_values(ascending=False)
    embedding = embedding.loc[col_sums.index, row_sums.index]

    fig, ax = plt.subplots(figsize=(x, y))

    my_cmap = matplotlib.colormaps['GnBu']
    my_cmap.set_under('w')
    ax.imshow(embedding.T, cmap=my_cmap, vmin=0.01, interpolation='nearest')
    ax.set_xticks(np.arange(embedding.shape[0]), labels=embedding.index, rotation=90, size=12)
    ax.set_yticks(np.arange(embedding.shape[1]), labels=embedding.columns, size=12)
    ax.set_xlabel('Criteria', size=14, weight='bold')
    ax.set_ylabel('Thematic metrics', size=14, weight='bold')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1 / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1 / 2))
    ax.tick_params(axis='both', which='minor', length=0)

    # add the labels across heatmap cells
    for i in range(len(embedding.index)):
        for j in range(len(embedding.columns)):
            if embedding.loc[embedding.index[i], embedding.columns[j]] > 0:
                ax.text(i, j, round(embedding.loc[embedding.index[i], embedding.columns[j]]),
                        ha="center", va="center", color='black', size=12, alpha=0.9)

    # Add row sums to the right of the heatmap
    for i, row_sum in enumerate(row_sums):
        ax.text(embedding.shape[0], i, round(row_sum), ha="center", va="center", color="black", size=12)

    # Add column sums below the heatmap
    for j, col_sum in enumerate(col_sums):
        ax.text(j, embedding.shape[1] - 6.7, round(col_sum), ha="center", va="bottom", color="black", size=12)

    ax.text(embedding.shape[1] + 10, embedding.shape[1] - 7.5, 'Occurrence count', ha="right", va="bottom",
            weight='bold', color="black", size=14)
    ax.text(embedding.shape[0] + 1, embedding.shape[0] - 25.5, 'Occurrence count', ha="center", rotation=90,
            weight='bold', va="center", color="black", size=14)

    ax.grid(which='minor', color='grey', linewidth=0.1)
    fig.tight_layout()

    return fig, ax


def plot_metric_embedding_heatmap(embedding_orig, x, y):
    """
    The plot displays the distribution of metrics that occur at least twice across existing criteria in the dataset.
    Args:
        embedding_orig: original embedding DataFrame.
        x: plot width.
        y: plot height. 
    """
    embedding = embedding_orig.copy().T
    row_sums = embedding.sum(axis=1).sort_values(ascending=True)
    col_sums = embedding.sum(axis=0).sort_values(ascending=False)

    embedding = embedding.loc[row_sums.index, col_sums.index]
    embedding = embedding.drop(columns=[col for col in embedding.columns if embedding.loc[:, col].sum() == 0])
    my_cmap = matplotlib.colormaps['GnBu']
    my_cmap.set_under('w')

    fig, ax = plt.subplots(figsize=(x, y))
    ax.imshow(embedding, aspect='auto', cmap=my_cmap, vmin=0.01, interpolation='nearest')
    ax.set_xticks(np.arange(embedding.shape[1]), labels=embedding.columns, rotation=90, size=15)
    ax.set_yticks(np.arange(embedding.shape[0]), labels=embedding.index, size=15)
    ax.set_ylabel('Frequent evaluation metrics', size=20)
    ax.set_xlabel('Frequent evaluation criteria', size=20)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1 / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1 / 2))
    ax.tick_params(axis='both', which='minor', length=0)

    prop_concentr = []

    test = embedding.loc[embedding.sum(axis=1) > 1, :]
    for i in test.index:
        prop_concentr.append(round((test.loc[i, :] / test.loc[i, :].sum(axis=0)).max(), 3))
    for i, row_sum in enumerate(row_sums):
        ax.text(embedding.shape[1], i + 0.15, prop_concentr[i], ha="left", va="bottom", color="black", size=15)

    # add the labels across heatmap cells
    for i in range(len(embedding.index)):
        for j in range(len(embedding.columns)):
            if embedding.loc[embedding.index[i], embedding.columns[j]] > 0:
                ax.text(j, i, round(embedding.loc[embedding.index[i], embedding.columns[j]]), ha="center", va="center",
                        color='black', alpha=0.9, size=13)

    ax.grid(which='minor', color='grey', linewidth=0.1)
    fig.tight_layout()

    return fig, ax


def plot_metric_criteria_occurrence_distribution(embedding, colors, x, y):
    """
    The plot displays the frequency and counts of the same metric linking to different criteria across all metrics.
    Args:
        embedding: array containing the metric and criteria relations.
        colors: color scheme for the plot.
    Returns:

    """
    metric_occurence = (embedding > 0).sum(axis=0).value_counts().sort_index()

    fig, ax = plt.subplots(1, 1, figsize=(x, y))
    bars = ax.bar(list(metric_occurence.index), metric_occurence, color=colors,
                  label=range(1, len(metric_occurence) + 1))

    ax.set_xlabel('No. of unique criteria', weight='bold', size=14)
    ax.set_ylabel('No. of metrics linked to criteria', weight='bold', size=14)
    ax.legend(fontsize=12)
    ax.grid(True, linewidth=0.3)

    # Iterating over the bars one-by-one
    for bar in bars:
        ax.annotate('{}'.format(bar.get_height()),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha='center', size=12, va='bottom')
    ax.xaxis.set_major_locator(plt.NullLocator())

    return fig, ax


def plot_aggregation_metric_distribution(metrics, colors, x, y):
    """
    The plot displays the frequency of different representation features across all metrics.
    Args:
        metrics: DataFrame containing the metrics and their properties.
        colors: color scheme for the plot.
    """
    f_metrics = {}
    f_method = {}

    for f_type in metrics['representation_feature'].dropna().unique():
        f_metrics[f_type] = len(metrics[metrics['representation_feature'] == f_type])
        f_method[f_type] = len(metrics[metrics['representation_feature'] == f_type]['method'].unique())

    fig, ax = plt.subplots(1, 1, figsize=(x, y))

    f_metrics = dict(sorted(f_metrics.items(), key=lambda item: item[1], reverse=True))
    bars = ax.bar(f_metrics.keys(), f_metrics.values(), color=colors, label=f_metrics.keys())
    ax.set_xlabel('Representation feature', weight='bold', size=14)
    ax.set_ylabel('No. of distinct metrics', weight='bold', size=14)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=12)

    # Iterating over the bars one-by-one
    for bar in bars:
        ax.annotate('{}'.format(bar.get_height()),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha='center', size=12, va='bottom')
    fig.tight_layout()

    return fig, ax


def plot_measurement_scale_distribution(metrics, colors, x, y):
    """
    The plot displays the frequency of different measurement scales across all metrics.
    Args:
        metrics: DataFrame containing the metrics and their properties.
        colors: color scheme for the plot.
    """
    m_metric = {}

    for m_type in metrics['measurement_scale'].dropna().unique():
        m_metric[m_type] = metrics.loc[metrics['measurement_scale'] == m_type, :].groupby(by='metric_type')[
            'method'].nunique().sum()
    fig, ax = plt.subplots(1, 1, figsize=(x, y))
    m_metric = dict(sorted(m_metric.items(), key=lambda item: item[1], reverse=True))
    bars = ax.bar(m_metric.keys(), m_metric.values(), color=colors, label=m_metric.keys())
    ax.set_xlabel('Measurement scales', size=14, weight='bold')
    ax.set_ylabel('No. of metrics', size=14, weight='bold')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.legend(fontsize=12)
    ax.grid(True, linewidth=0.3)

    # Iterating over the bars one-by-one
    for bar in bars:
        ax.annotate('{}'.format(bar.get_height()),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha='center', size=12, va='bottom')
    return fig, ax


def plot_scenario_1(metrics, features, thematic_metrics, criteria, labels, colors, x, y):
    """
    The plot illustrate the results of the scenario 1 metric selection strategy. The scenario 1 ensures even
    distribution of representation features.
    Args:
        labels: metric names that occurred at least twice in the metric dataset.
        colors: color scheme for the plot.
        x: plot width.
        y: plot height.
    """

    aggr_f = ['Node', 'Edge']
    aggr_metrics = metrics[metrics['representation_feature'].isin(aggr_f)]
    s1_metrics = set(aggr_metrics['metric_type'].unique())

    aggr_f_counts = features.copy()
    for f in aggr_f:
        aggr_f_counts[f] = aggr_metrics.loc[aggr_metrics['representation_feature'] == f, 'metric_type'].nunique()

    scenario = 1
    col_maximum = 25
    ylabels_rep = ['Representation Feature (goal focus)', 'Criteria', 'Thematic Metric']
    fig, axs = plt.subplots(1, 3, figsize=(x, y), tight_layout=True)
    fig.suptitle(f'Scenario {scenario}', size=16)
    for column in range(3):
        ax = axs[column]
        if column == 0:
            cur_values = aggr_f_counts.sort_index()
            bars = ax.barh(list(cur_values.index), cur_values, color='gray')
            for i, bar in enumerate(bars):
                if cur_values.iloc[i] > 0:
                    feature = cur_values.index[i]
                    bar.set_facecolor(colors[aggr_f.index(feature)])
        else:
            if column == 1:
                labels = criteria.sort_index().index
                counts = {}
                for feature in aggr_f:
                    cur_counts = criteria.sort_index().copy()
                    for crit in aggr_metrics['criteria_type'].dropna().unique():
                        cur_counts[crit] = aggr_metrics.loc[
                            (aggr_metrics['criteria_type'] == crit) & (
                                        aggr_metrics['representation_feature'] == feature),
                            'metric_type'
                        ].nunique()
                    counts[feature] = cur_counts
            elif column == 2:
                labels = thematic_metrics.sort_index().index
                counts = {}
                for feature in aggr_f:
                    cur_counts = thematic_metrics.sort_index().copy()
                    for theme in aggr_metrics['thematic_metric'].dropna().unique():
                        cur_counts[theme] = aggr_metrics.loc[
                            (aggr_metrics['thematic_metric'] == theme) & (
                                        aggr_metrics['representation_feature'] == feature),
                            'metric_type'
                        ].nunique()
                    counts[feature] = cur_counts
            left = np.zeros(len(labels))
            for feature, cur_counts in counts.items():
                bars = ax.barh(labels, cur_counts.to_numpy(), left=left, color=colors[aggr_f.index(feature)])
                left += cur_counts.to_numpy()
        ax.set_title(ylabels_rep[column], size=12)
        ax.set_xlabel('No. of metric instances', size=12)
        if column == 0:
            cur_max = cur_values.max()
        else:
            cur_max = left.max()
        ax.set_xlim(left=0, right=cur_max + cur_max * 0.1)
        ax.grid(True, linewidth=0.2)
        if column == 0:
            # Iterating over the bars one-by-one
            for bar in bars:
                if int(bar.get_width()) > 0:
                    ax.annotate('{}'.format(int(bar.get_width())),
                                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                                xytext=(2, 0), textcoords="offset points", ha='left', size=12, va='center')

    return s1_metrics, fig, axs


def plot_scenario_2(metrics, features, thematic_metrics, criteria, labels, colors, x, y):
    """
    The plot illustrate the results of the scenario 1 metric selection strategy. The scenario 1 ensures a balanced
    distribution of common evaluation criteria.
    Args:
        labels: metric names that occurred at least twice in the metric dataset.
        colors: color scheme for the plot.
        x: plot width.
        y: plot height.
    """
    criteria_index = metrics['criteria_type'].value_counts()[:5].index
    criteria_metrics = metrics[metrics['criteria_type'].isin(criteria_index)]
    s2_metrics = set(criteria_metrics['metric_type'].unique())

    criteria_counts = criteria.copy()
    for crit in criteria_index:
        criteria_counts[crit] = criteria_metrics.loc[criteria_metrics['criteria_type'] == crit, 'metric_type'].nunique()

    scenario = 2
    col_maximum = 30
    ylabels_rep = ['Criteria (goal focus)', 'Representation Feature', 'Thematic Metric']
    criteria_colors = dict(zip(criteria_index, colors[:5]))

    fig, axs = plt.subplots(1, 3, figsize=(x, y), tight_layout=True)
    fig.suptitle(f'Scenario {scenario}', size=16)
    relevant_criteria = []
    for column in range(3):
        ax = axs[column]
        if column == 0:
            cur_values = criteria_counts.sort_index()
            bars = ax.barh(list(cur_values.index), cur_values, color='gray')
            for i, bar in enumerate(bars):
                if cur_values.iloc[i] > 0:
                    criteria = cur_values.index[i]
                    bar.set_facecolor(criteria_colors[criteria])
                    relevant_criteria.append(criteria)
        else:
            if column == 1:
                labels = features.sort_index().index
                counts = {}
                for criteria in relevant_criteria:
                    cur_counts = features.sort_index().copy()
                    for feature in criteria_metrics['representation_feature'].dropna().unique():
                        cur_counts[feature] = criteria_metrics.loc[
                            (criteria_metrics['criteria_type'] == criteria) & (
                                        criteria_metrics['representation_feature'] == feature), 'metric_type'].nunique()
                    counts[criteria] = cur_counts
            elif column == 2:
                labels = thematic_metrics.sort_index().index
                counts = {}
                for criteria in relevant_criteria:
                    cur_counts = thematic_metrics.sort_index().copy()
                    for theme in criteria_metrics['thematic_metric'].dropna().unique():
                        cur_counts[theme] = criteria_metrics.loc[
                            (criteria_metrics['thematic_metric'] == theme) & (
                                        criteria_metrics['criteria_type'] == criteria),
                            'metric_type'
                        ].nunique()
                    counts[criteria] = cur_counts
            left = np.zeros(len(labels))
            for criteria, cur_counts in counts.items():
                bars = ax.barh(labels, cur_counts.to_numpy(), left=left, color=criteria_colors[criteria])
                left += cur_counts.to_numpy()
        ax.set_title(ylabels_rep[column], size=12)
        ax.set_xlabel('No. of metric instances', size=12)
        ax.set_xlim((0, col_maximum + 0.1 * col_maximum))
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if column == 0:
            cur_max = cur_values.max()
        else:
            cur_max = left.max()
        ax.set_xlim(left=0, right=cur_max + cur_max * 0.1)
        ax.grid(True, linewidth=0.2)
        if column == 0:
            # Iterating over the bars one-by-one
            for bar in bars:
                if int(bar.get_width()) > 0:
                    ax.annotate('{}'.format(int(bar.get_width())),
                                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                                xytext=(2, 0), textcoords="offset points", ha='left', size=12, va='center')

    return s2_metrics, fig, axs


def plot_scenario_3(metrics, features, thematic_metrics, criteria, labels, colors, x, y):
    """
    The plot illustrate the results of the scenario 1 metric selection strategy. The scenario 1 ensures a balanced
    distribution of metric themes.
    Args:
        labels: metric names that occurred at least twice in the metric dataset.
        colors: color scheme for the plot.
        x: plot width.
        y: plot height.
    """

    thematic_metrics_metrics = []
    thematic_metrics_counts = thematic_metrics.copy()

    for theme in thematic_metrics.index:
        metric_index = metrics[metrics['thematic_metric'] == theme]['metric_type'].value_counts()[:5].index
        cur_metrics = metrics.loc[metrics['metric_type'].isin(metric_index), :]
        thematic_metrics_metrics.append(cur_metrics)
        thematic_metrics_counts[theme] = cur_metrics['metric_type'].nunique()
    thematic_metrics_metrics = pd.concat(thematic_metrics_metrics, ignore_index=True)
    s3_metrics = set(thematic_metrics_metrics['metric_type'].unique())

    scenario = 3
    col_maximum = 30
    ylabels_rep = ['Thematic Metric (goal focus)', 'Representation Feature', 'Criteria']
    fig, axs = plt.subplots(1, 3, figsize=(x, y), tight_layout=True)
    fig.suptitle(f'Scenario {scenario}', size=16)
    for column in range(3):
        ax = axs[column]
        if column == 0:
            cur_values = thematic_metrics_counts.sort_index()
            bars = ax.barh(list(cur_values.index), cur_values, color='gray')
            for i, bar in enumerate(bars):
                bar.set_facecolor(colors[i])
        else:
            if column == 1:
                labels = features.sort_index().index
                counts = {}
                for theme in thematic_metrics_counts.sort_index().index:
                    cur_counts = features.sort_index().copy()
                    for feature in thematic_metrics_metrics['representation_feature'].dropna().unique():
                        cur_counts[feature] = thematic_metrics_metrics.loc[
                            (thematic_metrics_metrics['thematic_metric'] == theme) & (
                                        thematic_metrics_metrics['representation_feature'] == feature),
                            'metric_type'
                        ].nunique()
                    counts[theme] = cur_counts
            elif column == 2:
                labels = criteria.sort_index().index
                counts = {}
                for theme in thematic_metrics_counts.sort_index().index:
                    cur_counts = criteria.sort_index().copy()
                    for crit in thematic_metrics_metrics['criteria_type'].dropna().unique():
                        cur_counts[crit] = thematic_metrics_metrics.loc[
                            (thematic_metrics_metrics['thematic_metric'] == theme) & (
                                        thematic_metrics_metrics['criteria_type'] == crit), 'metric_type'].nunique()
                    counts[theme] = cur_counts
            left = np.zeros(len(labels))
            for theme, cur_counts in counts.items():
                bars = ax.barh(labels, cur_counts.to_numpy(), left=left,
                               color=colors[thematic_metrics_counts.sort_index().index.get_loc(theme)])
                left += cur_counts.to_numpy()
        ax.set_title(ylabels_rep[column], size=12)
        ax.set_xlabel('No. of metric instances', size=12)
        ax.set_xlim((0, col_maximum + 0.1 * col_maximum))
        ax.grid(True, linewidth=0.2)
        if column == 0:
            cur_max = cur_values.max()
        else:
            cur_max = left.max()
        ax.set_xlim(left=0, right=cur_max + cur_max * 0.1)
        if column == 0:
            # Iterating over the bars one-by-one
            for bar in bars:
                if int(bar.get_width()) > 0:
                    ax.annotate('{}'.format(int(bar.get_width())),
                                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                                xytext=(2, 0), textcoords="offset points", ha='left', size=12, va='center')
    return s3_metrics, fig, axs


def plot_overlapping_metrics(s1, s2, s3, colors, x, y):
    """
    Plot the overlapping metrics types between the three scenarios.
    Args:
        s1: set of metrics from scenario 1.
        s2: set of metrics from scenario 2.
        s3: set of metrics from scenario 3.
        colors: color scheme for the plot.
    """
    metric_names = list(s1 & s2 & s3)
    string = r"$\bf{Metric Type}$"
    for metric in metric_names:
        string += '\n' + metric
    fig, ax = plt.subplots(1, 1, figsize=(x, y), tight_layout=True)
    v = venn3([s1, s2, s3], (r"$\bf{S1}$", r"$\bf{S2}$", r"$\bf{S3}$"), colors, ax=ax)
    ax.annotate(string, xy=(-0.2, 0), xytext=(-200, -100),
                 ha='left', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='gray'))

    return fig, ax

