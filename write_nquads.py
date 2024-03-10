from metrics_to_nquads import TripleDataset
import os
import configparser
import pandas as pd


def main():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'paths.ini'))

    metric_df = pd.DataFrame(pd.read_excel(os.path.abspath(config['paths']['input_file'])))
    metric_dataset = TripleDataset()
    metric_dataset.create_metric_triples(metric_df)
    metric_dataset.write_triples(os.path.abspath(config['paths']['output_directory']))


if __name__ == "__main__":
    main()
