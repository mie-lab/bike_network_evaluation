import pandas as pd
from rdflib import Dataset, Literal, URIRef, XSD, RDF, RDFS
import uuid
import onto_manager as OM
import os

# column names from the input data
EVALUATION_METRIC = 'EvaluationMetric'
METRIC_TYPE = 'MetricType'
EVALUATION_METHOD = 'EvaluationMethod'
EVALUATION_CRITERION = 'EvaluationCriterion'
REPRESENTATION_FEATURE = 'RepresentationFeature'
SCORING_FUNCTION = 'ScoringFunction'
MEASUREMENT_SCALE = 'MeasurementScale'
UNIT = 'Unit'
UNIT_TYPE = 'UnitType'
FUNCTION = 'Function'
BUFFER = 'Buffer'
BUFFER_UNIT = 'BufferUnit'
COMMENT = 'Comment'
PARTS = 'Parts'

CARDINALITY_UNIT = 'CardinalityUnit' # refer to Foundational Global City Indicator Ontology.



class TripleDataset:

    def __init__(self):
        self.dataset = Dataset()

    def create_metric_triples(self, metrics):
        """
        creates triples to represent bike network evaluation metric used in an existing evaluation method (paper).
        Args:
            metrics: a metric with various associated properties described in a paper.
        """

        for metric in metrics.index:

            # main metric info
            metric_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            metric_type = URIRef(OM.PREFIX_NEMO + metrics.loc[metric, EVALUATION_METRIC])
            self.dataset.add((metric_uri, RDF.type, metric_type, OM.NEMO_GRAPH))
            # quantity
            self.dataset.add((metric_uri, RDF.type, OM.QUANTITY, OM.NEMO_GRAPH))
            measure_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            self.dataset.add((measure_uri, RDF.type, OM.MEASURE, OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.HAS_VALUE, measure_uri, OM.NEMO_GRAPH))
            self.dataset.add((measure_uri, OM.HAS_UNIT, measure_uri, OM.NEMO_GRAPH))
            # method
            method_uri = URIRef(metrics.loc[metric, EVALUATION_METHOD])
            self.dataset.add((method_uri, RDF.type, OM.EVALUATION_METHOD, OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.USED_IN, method_uri, OM.NEMO_GRAPH))

            # thematic metric
            if not pd.isnull(metrics.loc[metric, METRIC_TYPE]):
                general_metric_type = URIRef(OM.PREFIX_NEMO + metrics.loc[metric, METRIC_TYPE])
                self.dataset.add((metric_type, RDFS.subClassOf, general_metric_type, OM.NEMO_GRAPH))
            # criteria
            if not pd.isnull(metrics.loc[metric, EVALUATION_CRITERION]):
                self.create_criteria_triples(metrics.loc[metric, EVALUATION_CRITERION].split(','), metric_uri)
            # measurement scale
            if not pd.isnull(metrics.loc[metric, MEASUREMENT_SCALE]):
                self.create_measurement_scale_triples(metrics.loc[metric, MEASUREMENT_SCALE], metric_uri)
            # scoring scale
            if not pd.isnull(metrics.loc[metric, SCORING_FUNCTION]):
                self.create_scoring_scale_triples(metric_uri)
            # unit
            if not pd.isnull(metrics.loc[metric, UNIT]):
                self.create_unit_triples(metrics.loc[metric, UNIT], metrics.loc[metric, UNIT_TYPE], measure_uri)
            # aggregate unit function
            if not pd.isnull(metrics.loc[metric, FUNCTION]):
                self.create_aggregate_function_triples(metrics.loc[metric, FUNCTION], metric_uri)
            # composed metric parts
            if not pd.isnull(metrics.loc[metric, PARTS]):
                self.create_composed_metric_triples(metrics.loc[metric, PARTS].split(','), metric_uri)
            # aggregation feature
            if not pd.isnull(metrics.loc[metric, REPRESENTATION_FEATURE]):
                self.create_representation_feature_triples(metrics.loc[metric, REPRESENTATION_FEATURE], metric_uri)
            # buffered area
            if not pd.isnull(metrics.loc[metric, BUFFER]):
                buffer_list = str(metrics.loc[metric, BUFFER])
                self.create_buffer_triples(buffer_list.split(';'), metrics.loc[metric, BUFFER_UNIT], metric_uri)
            # comment
            if not pd.isnull(metrics.loc[metric, COMMENT]):
                comment = Literal(str(metrics.loc[metric, COMMENT]), datatype=XSD.string)
                self.dataset.add((metric_uri, RDFS.comment, comment, OM.NEMO_GRAPH))


    def create_representation_feature_triples(self, representation_feature, metric_uri):
        """
        creates triples that define on which spatial elements the metric value is aggregated (calculated).
        Args:
            representation_feature: a spatial feature,on which the relevant metric is calculated, e.g. network, edge, route.
            metric_uri: relevant metric uri.
        """
        res_feature_uri = URIRef(OM.PREFIX_NEMO + str(uuid.uuid1()))
        res_feature_class = URIRef(OM.PREFIX_NEMO + representation_feature)
        self.dataset.add((metric_uri, OM.MAPS_TO_FEATURE, res_feature_uri, OM.NEMO_GRAPH))
        self.dataset.add((res_feature_uri, RDF.type, res_feature_class, OM.NEMO_GRAPH))
        self.dataset.add((res_feature_uri, RDF.type, OM.REPRESENTATION_FEATURE, OM.NEMO_GRAPH))

    def create_criteria_triples(self, criterias, metric_uri):
        """
        creates triples defining bike network goodness criteria for which relevant metric is used.
        Args:
            criterias: A bike network goodness criteria, like safety, efficiency, quality, attractiveness.
            metric_uri: relevant metric uri.
        """
        for criteria in criterias:
            criteria_uri = URIRef(OM.PREFIX_NEMO + str(uuid.uuid1()))
            criteria_type = URIRef(OM.PREFIX_NEMO + criteria)
            self.dataset.add((criteria_uri, RDF.type, criteria_type, OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.MEASURES, criteria_uri, OM.NEMO_GRAPH))

    def create_unit_triples(self, units, unit_type, measure_uri):
        """
        creates unit triples for the relevant metric.
        Args:
            units: refer to OM ontology.
            unit_type: more general unit type compared to unit instance.
            measure_uri: relevant metric uri.
        """
        if unit_type == CARDINALITY_UNIT:
            unit_type_uri = URIRef(OM.PREFIX_GCI + unit_type)
        else:
            unit_type_uri = URIRef(OM.PREFIX_OM + unit_type)

        if len(units) == 3:
            pass
        elif len(units) == 2:
            combined_unit_uri = URIRef(OM.PREFIX_OM + units[0] + 'Per' + units[1])
            self.dataset.add((measure_uri, OM.HAS_UNIT, combined_unit_uri, OM.NEMO_GRAPH))
            self.dataset.add((combined_unit_uri, RDF.type, unit_type_uri, OM.NEMO_GRAPH))
        else:
            unit_uri = URIRef(OM.PREFIX_OM + units[0])
            self.dataset.add((measure_uri, OM.HAS_UNIT, unit_uri, OM.NEMO_GRAPH))
            self.dataset.add((unit_uri, RDF.type, unit_type_uri, OM.NEMO_GRAPH))

    def create_measurement_scale_triples(self, measurement_scale, metric_uri):
        """
        creates measuring mode triples that are linked to evaluation metric.
        Args:
            measurement_scale: the classical type of metric: nominal, ordinal, interval, ratio.
            metric_uri: relevant method uri
        """
        measurement_scale_uri = URIRef(OM.PREFIX_NEMO + measurement_scale)
        self.dataset.add((metric_uri, OM.HAS_MEASUREMENT_SCALE, measurement_scale_uri, OM.NEMO_GRAPH))
        self.dataset.add((measurement_scale_uri, RDF.type, OM.MEASUREMENT_SCALE, OM.NEMO_GRAPH))

    def create_scoring_scale_triples(self, metric_uri):
        """
        creates evaluation scale triples linked to the particular metric.
        Args:
            metric_uri: relevant metric uri.
        """
        scoring_scale_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
        self.dataset.add((metric_uri, OM.HAS_SCORING_FUNCTION, scoring_scale_uri, OM.NEMO_GRAPH))
        self.dataset.add((scoring_scale_uri, RDF.type, OM.SCORING_FUNCTION, OM.NEMO_GRAPH))

    def create_aggregate_function_triples(self, function, metric_uri):
        """
        creates aggregate function triples. It is used if metric value for example is an average.
        Args:
            function: a function can be applied to a quantity, like imn, max, average, for reference check OM ontology.
            metric_uri: relevant metric uri.
        """
        function_uri = URIRef(OM.PREFIX_OM + function)
        self.dataset.add((function_uri, RDF.type, OM.FUNCTION, OM.NEMO_GRAPH))
        self.dataset.add((metric_uri, OM.HAS_AGGREGATE_FUNCTION, function_uri, OM.NEMO_GRAPH))

    def create_composed_metric_triples(self, parts, metric_uri):
        """
        creates triples for metrics that is composed of multiple other metrics.
        Args:
            parts: other metrics from which the relevant metric is composed.
            metric_uri: relevant metric uri.
        """
        for part in parts:
            part_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            self.dataset.add((part_uri, RDF.type, URIRef(OM.PREFIX_NEMO + part), OM.NEMO_GRAPH))
            self.dataset.add((metric_uri, OM.COMPOSED_OF, part_uri, OM.NEMO_GRAPH))

    def create_buffer_triples(self, buffers, buffer_unit, metric_uri):
        """
        creates triples to define a buffered area around network element (edge or node)
        within which certain metrics are estimated.
        Args:
            buffers: distance from the edge or node.
            buffer_unit: unit that defines buffer extent, e.g. metre, minute.
            metric_uri: relevant metric uri.
        """
        for buffer in buffers:
            buffer_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            measure_uri = URIRef(OM.NEMO_GRAPH + str(uuid.uuid1()))
            # uncomment when interested in buffer value
            # buffer_value = Literal(str(buffer), datatype=XSD.integer)
            self.dataset.add((metric_uri, OM.MEASURED_WITHIN_BUFFER, buffer_uri, OM.NEMO_GRAPH))
            self.dataset.add((buffer_uri, RDF.type, buffer_uri, OM.NEMO_GRAPH))
            self.dataset.add((buffer_uri, OM.HAS_VALUE, measure_uri, OM.NEMO_GRAPH))
            # self.dataset.add((measure_uri, OM.HAS_NUMERIC_VALUE, buffer_value, OM.NEMO_GRAPH))
            self.dataset.add((measure_uri, OM.HAS_UNIT, URIRef(OM.PREFIX_OM + buffer_unit), OM.NEMO_GRAPH))

    def write_triples(self, out_dir):
        """
        Writes generated triples to a file.
        Args:
            out_dir: directory where the file will be saved.
        """

        with open(os.path.join(out_dir, "metrics.nq"), mode="w+") as file:
            file.write(self.dataset.serialize(format='nquads'))