import pandas as pd


class DataFrameContainer:
    """
    Holds two DataFrames: 'training' and 'test'.
    Provides get/set accessors and a method to add columns.
    """

    def __init__(self, training: pd.DataFrame, test: pd.DataFrame):
        self._frames = {
            'training': training.copy(),
            'test': test.copy()
        }

    def get(self, which: str) -> pd.DataFrame:
        """
        Return the requested DataFrame.

        Parameters:
            which (str): 'training' or 'test'

        Returns:
            pd.DataFrame
        """
        if which not in self._frames:
            raise ValueError("which must be 'training' or 'test'")
        return self._frames[which]

    def set(self, which: str, new_df: pd.DataFrame) -> None:
        """
        Replace one of the DataFrames with a new one.

        Parameters:
            which (str): 'training' or 'test'
            new_df (pd.DataFrame): the DataFrame to set
        """
        if which not in self._frames:
            raise ValueError("which must be 'training' or 'test'")
        self._frames[which] = new_df.copy()

    def add_column(
            self,
            df_name: str,
            col_name: str,
            data,
            both: bool = False
    ) -> None:
        """
        Add a column to one (or both) DataFrames.

        Parameters:
            df_name (str): either 'training' or 'test'; ignored if both=True
            col_name (str): the new column name
            data: scalar or array-like matching the DataFrame length
            both (bool): if True, add to both training & test
        """
        targets = ['training', 'test'] if both else [df_name]
        for key in targets:
            if key not in self._frames:
                raise ValueError("df_name must be 'training' or 'test'")
            df = self._frames[key]
            df[col_name] = data