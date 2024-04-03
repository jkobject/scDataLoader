from collections import Counter
from functools import reduce
from os import PathLike
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from lamin_utils import logger
from lamindb.dev._data import _track_run_input
from lamindb.dev.storage._backed_access import (
    ArrayTypes,
    GroupTypes,
    StorageType,
    _safer_read_index,
    registry,
)
from lamindb_setup.dev.upath import UPath


class _Connect:
    def __init__(self, storage):
        if isinstance(storage, UPath):
            self.conn, self.store = registry.open("h5py", storage)
            self.to_close = True
        else:
            self.conn, self.store = None, storage
            self.to_close = False

    def __enter__(self):
        return self.store

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if not self.to_close:
            return
        if hasattr(self.store, "close"):
            self.store.close()
        if hasattr(self.conn, "close"):
            self.conn.close()


def mapped(
    dataset,
    stream: bool = False,
    is_run_input: Optional[bool] = None,
    **kwargs,
) -> "MappedDataset":
    _track_run_input(dataset, is_run_input)
    path_list = []
    for file in dataset.artifacts.all():
        if file.suffix not in {".h5ad", ".zrad", ".zarr"}:
            logger.warning(f"Ignoring file with suffix {file.suffix}")
            continue
        elif not stream and file.suffix == ".h5ad":
            path_list.append(file.stage())
        else:
            path_list.append(file.path)
    return MappedDataset(path_list, **kwargs)


class MappedDataset:
    """Map-style dataset for use in data loaders.

    This currently only works for collections of `AnnData` objects.

    For an example, see :meth:`~lamindb.Dataset.mapped`.

    .. note::

        A similar data loader exists `here
        <https://github.com/Genentech/scimilarity>`__.
    """

    def __init__(
        self,
        path_list: List[Union[str, PathLike]],
        label_keys: Optional[Union[str, List[str]]] = None,
        join_vars: Optional[Literal["auto", "inner", "None"]] = "auto",
        encode_labels: Optional[Union[bool, List[str]]] = False,
        parallel: bool = False,
        unknown_class: str = "unknown",
    ):
        self.storages = []
        self.conns = []
        self.parallel = parallel
        self.unknown_class = unknown_class
        self.path_list = path_list
        self._make_connections(path_list, parallel)

        self.n_obs_list = []
        for storage in self.storages:
            with _Connect(storage) as store:
                X = store["X"]
                index = (
                    store["var"]["ensembl_gene_id"]
                    if "ensembl_gene_id" in store["var"]
                    else store["var"]["_index"]
                )
                if join_vars == "None":
                    if not all(
                        [
                            i <= j
                            for i, j in zip(
                                index[:99],
                                index[1:100],
                            )
                        ]
                    ):
                        raise ValueError("The variables are not sorted.")
                if isinstance(X, ArrayTypes):  # type: ignore
                    self.n_obs_list.append(X.shape[0])
                else:
                    self.n_obs_list.append(X.attrs["shape"][0])
        self.n_obs = sum(self.n_obs_list)

        self.indices = np.hstack([np.arange(n_obs) for n_obs in self.n_obs_list])
        self.storage_idx = np.repeat(np.arange(len(self.storages)), self.n_obs_list)

        self.join_vars = join_vars if len(path_list) > 1 else None
        self.var_indices = None
        if self.join_vars != "None":
            self._make_join_vars()

        self.encode_labels = encode_labels
        self.label_keys = [label_keys] if isinstance(label_keys, str) else label_keys
        if isinstance(encode_labels, bool):
            if encode_labels:
                encode_labels = label_keys
            else:
                encode_labels = []
        if isinstance(encode_labels, list):
            self.encoders = {}
            for label in encode_labels:
                cats = self.get_merged_categories(label)
                self.encoders[label] = {cat: i for i, cat in enumerate(cats)}
                if unknown_class in self.encoders[label]:
                    self.encoders[label][unknown_class] = -1
        else:
            self.encoders = {}
        self._closed = False

    def _make_connections(self, path_list: list, parallel: bool):
        for path in path_list:
            path = UPath(path)
            if path.exists() and path.is_file():  # type: ignore
                if parallel:
                    conn, storage = None, path
                else:
                    conn, storage = registry.open("h5py", path)
            else:
                conn, storage = registry.open("zarr", path)
            self.conns.append(conn)
            self.storages.append(storage)

    def _make_join_vars(self):
        var_list = []
        for storage in self.storages:
            with _Connect(storage) as store:
                var_list.append(_safer_read_index(store["var"]))
        if self.join_vars == "auto":
            vars_eq = all([var_list[0].equals(vrs) for vrs in var_list[1:]])
            if vars_eq:
                self.join_vars = None
                return
            else:
                self.join_vars = "inner"
        if self.join_vars == "inner":
            self.var_joint = reduce(pd.Index.intersection, var_list)
            if len(self.var_joint) == 0:
                raise ValueError(
                    "The provided AnnData objects don't have shared varibales."
                )
            self.var_indices = [vrs.get_indexer(self.var_joint) for vrs in var_list]

    def _check_aligned_vars(self, vars: list):
        i = 0
        for storage in self.storages:
            with _Connect(storage) as store:
                if vars == _safer_read_index(store["var"]).tolist():
                    i += 1
        print("{}% are aligned".format(i * 100 / len(self.storages)))

    def __len__(self):
        return self.n_obs

    def __getitem__(self, idx: int):
        obs_idx = self.indices[idx]
        storage_idx = self.storage_idx[idx]
        if self.var_indices is not None:
            var_idxs = self.var_indices[storage_idx]
        else:
            var_idxs = None
        with _Connect(self.storages[storage_idx]) as store:
            out = {"x": self.get_data_idx(store, obs_idx, var_idxs)}
            if self.label_keys is not None:
                for _, label in enumerate(self.label_keys):
                    label_idx = self.get_label_idx(store, obs_idx, label)
                    if label in self.encoders:
                        out.update({label: self.encoders[label][label_idx]})
                    else:
                        out.update({label: label_idx})
                out.update({"dataset": storage_idx})
        return out

    def get_data_idx(
        self,
        storage: StorageType,
        idx: int,
        var_idxs: Optional[list] = None,
        layer_key: Optional[str] = None,  # type: ignore # noqa
    ):
        """Get the index for the data."""
        layer = storage["X"] if layer_key is None else storage["layers"][layer_key]  # type: ignore # noqa
        if isinstance(layer, ArrayTypes):  # type: ignore
            # todo: better way to select variables

            return layer[idx] if var_idxs is None else layer[idx][var_idxs]
        else:  # assume csr_matrix here
            data = layer["data"]
            indices = layer["indices"]
            indptr = layer["indptr"]
            s = slice(*(indptr[idx : idx + 2]))
            # this requires more memory than csr_matrix when var_idxs is not None
            # but it is faster
            layer_idx = np.zeros(layer.attrs["shape"][1])
            layer_idx[indices[s]] = data[s]
            return layer_idx if var_idxs is None else layer_idx[var_idxs]

    def get_label_idx(self, storage: StorageType, idx: int, label_key: str):  # type: ignore # noqa
        """Get the index for the label by key."""
        obs = storage["obs"]  # type: ignore
        # how backwards compatible do we want to be here actually?
        if isinstance(obs, ArrayTypes):  # type: ignore
            label = obs[idx][obs.dtype.names.index(label_key)]
        else:
            labels = obs[label_key]
            if isinstance(labels, ArrayTypes):  # type: ignore
                label = labels[idx]
            else:
                label = labels["codes"][idx]

        cats = self.get_categories(storage, label_key)
        if cats is not None:
            label = cats[label]
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        return label

    def get_label_weights(self, label_keys: Union[str, List[str]], scaler=10):
        """Get all weights for a given label key."""
        if type(label_keys) is not list:
            label_keys = [label_keys]
        for i, val in enumerate(label_keys):
            if val not in self.label_keys:
                raise ValueError(f"{val} is not a valid label key.")
            if i == 0:
                labels = self.get_merged_labels(val)
            else:
                labels += "_" + self.get_merged_labels(val).astype(str).astype("O")
        counter = Counter(labels)  # type: ignore
        rn = {n: i for i, n in enumerate(counter.keys())}
        labels = np.array([rn[label] for label in labels])
        counter = np.array(list(counter.values()))
        weights = scaler / (counter + scaler)
        return weights, labels

    def get_merged_labels(self, label_key: str):
        """Get merged labels."""
        labels_merge = []
        decode = np.frompyfunc(lambda x: x.decode("utf-8"), 1, 1)
        for storage in self.storages:
            with _Connect(storage) as store:
                codes = self.get_codes(store, label_key)
                labels = decode(codes) if isinstance(codes[0], bytes) else codes
                cats = self.get_categories(store, label_key)
                if cats is not None:
                    cats = decode(cats) if isinstance(cats[0], bytes) else cats
                    labels = cats[labels]
                labels_merge.append(labels)
        return np.hstack(labels_merge)

    def get_merged_categories(self, label_key: str):
        """Get merged categories."""
        cats_merge = set()
        decode = np.frompyfunc(lambda x: x.decode("utf-8"), 1, 1)
        for storage in self.storages:
            with _Connect(storage) as store:
                cats = self.get_categories(store, label_key)
                if cats is not None:
                    cats = decode(cats) if isinstance(cats[0], bytes) else cats
                    cats_merge.update(cats)
                else:
                    codes = self.get_codes(store, label_key)
                    codes = decode(codes) if isinstance(codes[0], bytes) else codes
                    cats_merge.update(codes)
        return cats_merge

    def get_categories(self, storage: StorageType, label_key: str):  # type: ignore
        """Get categories."""
        obs = storage["obs"]  # type: ignore
        if isinstance(obs, ArrayTypes):  # type: ignore
            cat_key_uns = f"{label_key}_categories"
            if cat_key_uns in storage["uns"]:  # type: ignore
                return storage["uns"][cat_key_uns]  # type: ignore
            else:
                return None
        else:
            if "__categories" in obs:
                cats = obs["__categories"]
                if label_key in cats:
                    return cats[label_key]
                else:
                    return None
            labels = obs[label_key]
            if isinstance(labels, GroupTypes):  # type: ignore
                if "categories" in labels:
                    return labels["categories"]
                else:
                    return None
            else:
                if "categories" in labels.attrs:
                    return labels.attrs["categories"]
                else:
                    return None

    def get_codes(self, storage: StorageType, label_key: str):  # type: ignore
        """Get codes."""
        obs = storage["obs"]  # type: ignore
        if isinstance(obs, ArrayTypes):  # type: ignore
            label = obs[label_key]
        else:
            label = obs[label_key]
            if isinstance(label, ArrayTypes):  # type: ignore
                return label[...]
            else:
                return label["codes"][...]

    def close(self):
        """Close connection to array streaming backend."""
        for storage in self.storages:
            if hasattr(storage, "close"):
                storage.close()
        for conn in self.conns:
            if hasattr(conn, "close"):
                conn.close()
        self._closed = True

    @property
    def closed(self):
        return self._closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
