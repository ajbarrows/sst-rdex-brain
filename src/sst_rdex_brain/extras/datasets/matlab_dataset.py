
from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import pandas as pd
from scipy.io import loadmat

from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path

class MatlabDataSet(AbstractDataSet[pd.DataFrame, pd.DataFrame]):

    def __init__(self, filepath: str):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self.prefix = filepath.replace('.mat', '').split('/')[-1]


    def _load(self) -> pd.DataFrame:
        load_path = get_filepath_str(self._filepath, self._protocol)
        
        mat_dict = loadmat(load_path)
        key = list(mat_dict)[-1]
        df = pd.DataFrame(mat_dict[key].T)
        df = df.add_prefix(self.prefix + '_')
        
        return df
    
    def _save(self, data: pd.DataFrame) -> None:
        ...
    
    def _describe(self) -> Dict[str, Any]:
        ...