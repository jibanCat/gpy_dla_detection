import pandas as pd


class QuasarTableLoader:
    def __init__(
        self,
        filepath="data/lls_catalogs/staa2388_supplemental_table_data/table_data_full.txt",
    ):
        self.filepath = filepath
        self.column_names = [
            "quasar_name",
            "right_ascension_deg",
            "declination_deg",
            "redshift",
            "SN_1150A",
            "science_primary",
            "in_training_set",
            "classification_outcome",
            "LLS_redshift",
        ]

    def load_data(self, sep="\s+"):
        """
        Loads the quasar data from the specified file and saves each column's values as an attribute of the class.

        :param sep: The delimiter between columns in the file.
        """
        try:
            # Load the data into a DataFrame
            self.df = pd.read_csv(
                self.filepath,
                sep=sep,
                names=self.column_names,
                skiprows=list(range(15)),  # Skip the first 15 rows
            )

            # Assign each column's values to an attribute of the class
            for column in self.column_names:
                setattr(self, column, self.df[column])

        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            self.df = None  # Ensure df is set to None in case of failure
