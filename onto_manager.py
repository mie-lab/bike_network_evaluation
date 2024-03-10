from rdflib import URIRef

PREFIX_NEMO = 'http://www.ebikecityevaluationtool.com/ontology/nemo#'
PREFIX_OM = 'http://www.ontology-of-units-of-measure.org/resource/om-2/'
PREFIX_GEO = 'http://www.opengis.net/ont/geosparql#'
PREFIX_GCI = 'http://ontology.eil.utoronto.ca/GCI/Foundation/GCI-Foundation.owl#'
NEMO_GRAPH = URIRef('http://www.ebikecityevaluationtool.com/ontology/nemo/metrics/')

# VeloNEMO ontology
EVALUATION_CRITERION = URIRef(PREFIX_NEMO + 'EvaluationCriterion')
SCORING_FUNCTION = URIRef(PREFIX_NEMO + 'ScoringFunction')
EVALUATION_METHOD = URIRef(PREFIX_NEMO + 'EvaluationMethod')
WEIGHT = URIRef(PREFIX_NEMO + 'Weight')
REPRESENTATION_FEATURE = URIRef(PREFIX_NEMO + 'RepresentationFeature')
EVALUATION_METRIC = URIRef(PREFIX_NEMO + 'EvaluationMetric')
MEASUREMENT_SCALE = URIRef(PREFIX_NEMO + 'MeasurementScale')
DATA_SOURCE = URIRef(PREFIX_NEMO + 'DataSource')
COMPOSITE_METRIC = URIRef(PREFIX_NEMO + 'CompositeMetric')
CONTEXTUAL_METRIC = URIRef(PREFIX_NEMO + 'ContextualMetric')
INFRASTRUCTURAL_METRIC = URIRef(PREFIX_NEMO + 'InfrastructuralMetric')
MORPHOLOGICAL_METRIC = URIRef(PREFIX_NEMO + 'MorphologicalMetric')
MODE_METRIC = URIRef(PREFIX_NEMO + 'ModeMetric')
TOPOLOGICAL_METRIC = URIRef(PREFIX_NEMO + 'GraphMetric')

# Nemo properties
HAS_SCORING_FUNCTION = URIRef(PREFIX_NEMO + 'hasScoringFunction')
MAPS_TO_FEATURE = URIRef(PREFIX_NEMO + 'mapsToFeature')
HAS_MEASUREMENT_SCALE = URIRef(PREFIX_NEMO + 'hasMeasurementScale')
USED_IN = URIRef(PREFIX_NEMO + 'usedIn')
ASSESSED_IN = URIRef(PREFIX_NEMO + 'AssessedIn')
MEASURES = URIRef(PREFIX_NEMO + 'measures')
WEIGHTED_BY = URIRef(PREFIX_NEMO + 'weightedBy')
DEFINED_BY = URIRef(PREFIX_NEMO + 'definedBy')
MEASURED_WITHIN_BUFFER = URIRef(PREFIX_NEMO + 'measuredWithinBuffer')
COMPOSED_OF = URIRef(PREFIX_NEMO + 'composedOf')
HAS_SOURCE = URIRef(PREFIX_NEMO + 'hasSource')

# OM ontology
FUNCTION = URIRef(PREFIX_OM + 'Function')
MEASURE = URIRef(PREFIX_OM + 'Measure')
QUANTITY = URIRef(PREFIX_OM + 'Quantity')
UNIT = URIRef(PREFIX_OM + 'Unit')
HAS_AGGREGATE_FUNCTION = URIRef(PREFIX_OM + 'hasAggregateFunction')
HAS_UNIT = URIRef(PREFIX_OM + 'hasUnit')
HAS_VALUE = URIRef(PREFIX_OM + 'hasValue')
HAS_NUMERIC_VALUE = URIRef(PREFIX_OM + 'hasNumericValue')

# geoSPARQL ontology
FEATURE = URIRef(PREFIX_GEO + 'Feature')
GEOMETRY = URIRef(PREFIX_GEO + 'Geometry')
HAS_GEOMETRY = URIRef(PREFIX_GEO + 'hasGeometry')
AS_WKT = URIRef(PREFIX_GEO + 'asWKT')

# global city indicator ontology
CARDINALITY_UNIT = URIRef(PREFIX_GCI + 'Cardinality_unit')


