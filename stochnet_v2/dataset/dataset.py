import abc
import h5py
import logging
import numpy as np
import os
import pickle
import tensorflow as tf

from bidict import bidict
from sklearn.preprocessing.data import StandardScaler, MinMaxScaler
from tqdm import tqdm
from stochnet_v2.utils.errors import ShapeError

LOGGER = logging.getLogger('dataset.dataset')


class DataTransformer:
    """DataTransformer transforms CRN traces into training examples with optional scaling."""

    def __init__(
            self,
            dataset_address,
            with_timestamps=True,
            nb_randomized_params=0,
    ):
        """
        Initialize transformer.

        Parameters
        ----------
        dataset_address : filepath to the dataset containing CRN traces.
            Data in the file should be of shape [nb_traces, nb_steps, nb_features].
            If with_timestamps is True, the first feature is considered as time.
        with_timestamps : boolean, whether time is included in data (as the very first feature)
            Data produced by scripts/simulate_data_gillespy.py has time, therefore default values is True.
        """
        self.nb_trajectories = None
        self.nb_timesteps = None
        self.nb_features = None
        self.labels = None
        self.with_labels = False
        self.with_timestamps = with_timestamps
        self.nb_randomized_params = nb_randomized_params

        self._scaler = None
        self.scaler_is_fitted = False
        self.scaler_positivity = None
        self.dtype = np.float32

        self.read_data(dataset_address)

    @property
    def scaler(self):
        return self._scaler

    def read_data(self, dataset_address):
        """Read data and memorize shape."""
        with open(dataset_address, 'rb') as data_file:
            self.data = np.asarray(np.load(data_file), dtype=self.dtype)
            self._memorize_dataset_shape()

    def _memorize_dataset_shape(self):
        """Memorize data shape."""
        if self.data.ndim != 3:
            raise ShapeError(
                f"The dataset is not properly formatted.\n"
                f"We expect the following shape: "
                f"(nb_trajectories, nb_timesteps, nb_features),\n"
                f"got: {self.data.shape}"
            )
        self.nb_trajectories, self.nb_timesteps, self.nb_features = self.data.shape

    def set_labels(self, labels):
        """
        Set labels for species.

        Parameters
        ----------
        labels : list of species names. Length of the list and the order of names
            should coincide with the species presented in data (excluding `time`).

        """
        if labels is None:
            self.labels = None
            self.with_labels = False
        else:
            if self.with_timestamps:
                labels = ['timestamp'] + labels

            if len(labels) != self.nb_features:
                raise ShapeError(
                    f"There needs to be exactly one label for each feature.\n"
                    f"We have {len(labels)} labels for {self.nb_features} features."
                )
            self.labels = bidict(zip(range(len(labels)), labels))
            self.with_labels = True

    def drop_timestamps(self):
        """Drop time from data."""
        if self.with_timestamps is True:

            self.data = self.data[..., 1:]
            self.nb_features = self.nb_features - 1
            self.with_timestamps = False
            self._memorize_dataset_shape()

            if self.with_labels is True:
                self.labels.inv.pop('timestamp')
                self.labels = bidict(zip(
                    [k - 1 for k in self.labels.keys()],
                    self.labels.values()
                ))

    def _create_scaler(self, positivity):
        self.scaler_positivity = positivity
        if positivity is True:
            eps = 1e-9
            self._scaler = MinMaxScaler(feature_range=(eps, 1))
        else:
            self._scaler = StandardScaler()
        self.scaler_is_fitted = False

    def _fit_scaler(self, positivity=False, slice_size=None):
        if (self._scaler is None) or (self.scaler_positivity != positivity):
            self._create_scaler(positivity)

        if not self.scaler_is_fitted:

            LOGGER.info(f"Fitting scaler, positivity={positivity}")

            if slice_size is None:
                self._scaler.fit(self.data.reshape(-1, self.nb_features))
            else:
                n_slices = self.nb_trajectories // slice_size

                for i in tqdm(range(n_slices)):
                    data_slice = self.data[i * slice_size: (i + 1) * slice_size, ...]
                    data_slice = data_slice.reshape(-1, self.nb_features)
                    self._scaler.partial_fit(data_slice)

                if self.nb_trajectories % slice_size != 0:
                    data_slice = self.data[n_slices * slice_size:, ...]
                    data_slice = data_slice.reshape(-1, self.nb_features)
                    self._scaler.partial_fit(data_slice)

            self.scaler_is_fitted = True

    def rescale(self, data):
        """
        Apply scaler to data.

        Parameters
        ----------
        data : data to rescale.

        Returns
        -------
        data : rescaled data.

        """
        # return self.scaler.transform(data)
        if isinstance(self.scaler, StandardScaler):
            try:
                data = (data - self.scaler.mean_) / self.scaler.scale_
            except ValueError:
                data = (data - self.scaler.mean_[:-self.nb_randomized_params]) \
                       / self.scaler.scale_[:-self.nb_randomized_params]
        elif isinstance(self.scaler, MinMaxScaler):
            try:
                data = (data * self.scaler.scale_) + self.scaler.min_
            except ValueError:
                data = (data * self.scaler.scale_[:-self.nb_randomized_params]) \
                       + self.scaler.min_[:-self.nb_randomized_params]
        return data

    def scale_back(self, data):
        """
        Apply scaler inverse transform, returning data to the original scale.
        Parameters
        ----------
        data : data (rescaled).

        Returns
        -------
        data : data scaled back.

        """
        # return self.scaler.inverse_transform(data)
        if isinstance(self.scaler, StandardScaler):
            try:
                data = data * self.scaler.scale_ + self.scaler.mean_
            except ValueError:
                data = data * self.scaler.scale_[:-self.nb_randomized_params] \
                       + self.scaler.mean_[:-self.nb_randomized_params]
        elif isinstance(self.scaler, MinMaxScaler):
            try:
                data = (data - self.scaler.min_) / self.scaler.scale_
            except ValueError:
                data = (data - self.scaler.min_[:-self.nb_randomized_params]) \
                       / self.scaler.scale_[:-self.nb_randomized_params]
        return data

    def _shuffle_data(self):
        np.random.shuffle(self.data)

    def _transitions_from_a_batch_of_trajectories(
            self,
            trajectories,
            nb_past_timesteps,
    ):
        x_data = []
        y_data = []

        for timestep in range(self.nb_timesteps - nb_past_timesteps):
            x_data.append(trajectories[:, timestep: (timestep + nb_past_timesteps), :])
            y_data.append(trajectories[:, timestep + nb_past_timesteps, :])

        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        return x_data, y_data

    def _transitions_generator(
            self,
            trajectories,
            nb_past_timesteps,
            slice_size=None,
            rescale=False,
    ):
        self._check_nb_past_timesteps(nb_past_timesteps)
        nb_trajectories = trajectories.shape[0]

        if slice_size:
            n_slices = nb_trajectories // slice_size
            additive = 0 if nb_trajectories % slice_size == 0 else 1
        else:
            n_slices = 1
            additive = 0
            slice_size = nb_trajectories

        for i in range(n_slices + additive):
            if i == n_slices:
                x_data, y_data = self._transitions_from_a_batch_of_trajectories(
                    trajectories[slice_size * n_slices: nb_trajectories],
                    nb_past_timesteps,
                )
            else:
                x_data, y_data = self._transitions_from_a_batch_of_trajectories(
                    trajectories[slice_size * i: slice_size * (i + 1)],
                    nb_past_timesteps,
                )
            if rescale:
                x_data = self.rescale(x_data)
                y_data = self.rescale(y_data)

            yield x_data, y_data

    def _train_test_generators(
            self,
            nb_past_timesteps,
            test_fraction=0.2,
            slice_size=None,
            rescale=False,
    ):
        n_train_trajectories = int((1. - test_fraction) * self.nb_trajectories)

        train_gen = self._transitions_generator(
            self.data[:n_train_trajectories],
            nb_past_timesteps,
            slice_size,
            rescale,
        )
        test_gen = self._transitions_generator(
            self.data[n_train_trajectories:],
            nb_past_timesteps,
            slice_size,
            rescale,
        )
        return train_gen, test_gen

    def get_train_test_data_generators(
            self,
            nb_past_timesteps=1,
            test_fraction=0.2,
            keep_timestamps=False,
            rescale=True,
            positivity=True,
            shuffle=True,
            slice_size=None,

    ):
        """
        Produce data generators, yielding chunks of transformed data,
        containing (optionally) rescaled training examples.
        Each training example is a single transition between states of the system:
            (x, y) = (trajectory[i:i+nb_past_timesteps], trajectory[i+nb_past_timesteps])

        Parameters
        ----------
        nb_past_timesteps : number of steps observed before each transition.
        test_fraction : float, fraction of data that will be used for test.
        keep_timestamps : boolean, whether to keep timestamps in data, default is False.
        rescale : boolean, whether data should be rescaled.
        positivity : boolean, if True, data will be rescaled between 0 and 1, otherwise standardized.
        shuffle : boolean, if True trajectories will be shuffled before producing training examples.
        slice_size : int, number of trajectories to process at once, optional. May be useful for
            large datasets to reduce memory consumption. If None, all trajectories used.

        Returns
        -------
        (train_generator, test_generator) : iterable generators of training examples.
            Every iteration of generator yields training examples produced from
            `slice_size` number of trajectories.
        """
        if keep_timestamps is False:
            self.drop_timestamps()

        if rescale is True:
            self._fit_scaler(positivity, slice_size)

        if shuffle is True:
            self._shuffle_data()

        return self._train_test_generators(
            nb_past_timesteps,
            test_fraction,
            slice_size,
            rescale,
        )

    def _check_nb_past_timesteps(self, nb_past_timesteps):
        if nb_past_timesteps + 1 > self.nb_timesteps:
            raise ValueError('Too many past timesteps.')
        elif nb_past_timesteps < 1:
            raise ValueError('You need to consider at least 1 timestep in the past.')

    def _save_scaler(self, dataset_folder):
        scaler_fp = os.path.join(dataset_folder, 'scaler.pickle')
        with open(scaler_fp, 'wb') as file:
            pickle.dump(self.scaler, file)

    def save_data_for_ml_hdf5(
            self,
            dataset_folder,
            nb_past_timesteps=1,
            test_fraction=0.2,
            keep_timestamps=False,
            rescale=True,
            positivity=True,
            shuffle=True,
            slice_size=None,
            force_rewrite=False,
    ):
        """
        Write training and test datasets to hdf5 files.
        Original trajectories are optionally scaled and split into training examples:
            (x, y) = (trajectory[i:i+nb_past_timesteps], trajectory[i+nb_past_timesteps])

        Parameters
        ----------
        dataset_folder : folder to save datasets
        nb_past_timesteps : number of steps observed before each transition.
        test_fraction : float, fraction of data that will be used for test.
        keep_timestamps : boolean, whether to keep timestamps in data, default is False.
        rescale : boolean, whether data should be rescaled.
        positivity : boolean, if True, data will be rescaled between 0 and 1, otherwise standardized.
        shuffle : boolean, if True trajectories will be shuffled before producing training examples.
        slice_size : int, number of trajectories to process at once, optional. May be useful for
            large datasets to reduce memory consumption. If None, all trajectories used.
        force_rewrite : boolean, if True, existing files will be rewritten.

        Returns
        -------
        None

        """
        train_gen, test_gen = self.get_train_test_data_generators(
            nb_past_timesteps=nb_past_timesteps,
            test_fraction=test_fraction,
            keep_timestamps=keep_timestamps,
            rescale=rescale,
            positivity=positivity,
            shuffle=shuffle,
            slice_size=slice_size,
        )

        if rescale:
            self._save_scaler(dataset_folder)
            train_fp = os.path.join(dataset_folder, 'train_rescaled.hdf5')
            test_fp = os.path.join(dataset_folder, 'test_rescaled.hdf5')
        else:
            train_fp = os.path.join(dataset_folder, 'train.hdf5')
            test_fp = os.path.join(dataset_folder, 'test.hdf5')

        if force_rewrite:
            if os.path.exists(train_fp):
                os.remove(train_fp)
            if os.path.exists(test_fp):
                os.remove(test_fp)

        with h5py.File(train_fp, 'a', libver='latest') as df:
            df.create_dataset(
                'x',
                shape=(0, nb_past_timesteps, self.nb_features),
                maxshape=(None, nb_past_timesteps, self.nb_features),
                chunks=True,
            )
            df.create_dataset(
                'y',
                shape=(0, self.nb_features - self.nb_randomized_params),
                maxshape=(None, self.nb_features - self.nb_randomized_params),
                chunks=True,
            )
            for x, y in train_gen:
                n_new_items = x.shape[0]
                df['x'].resize(df['x'].shape[0] + n_new_items, axis=0)
                df['x'][-n_new_items:] = x
                
                df['y'].resize(df['y'].shape[0] + n_new_items, axis=0)
                df['y'][-n_new_items:] = y[..., :-self.nb_randomized_params]

            LOGGER.info(f"Train data saved to {train_fp}, \n"
                        f"Shapes: x: {df['x'].shape}, y: {df['y'].shape}")

        with h5py.File(test_fp, 'a', libver='latest') as df:
            df.create_dataset(
                'x',
                shape=(0, nb_past_timesteps, self.nb_features),
                maxshape=(None, nb_past_timesteps, self.nb_features),
                chunks=True,
            )
            df.create_dataset(
                'y',
                shape=(0, self.nb_features - self.nb_randomized_params),
                maxshape=(None, self.nb_features - self.nb_randomized_params),
                chunks=True,
            )
            for x, y in test_gen:
                n_new_items = x.shape[0]
                df['x'].resize(df['x'].shape[0] + n_new_items, axis=0)
                df['x'][-n_new_items:] = x

                df['y'].resize(df['y'].shape[0] + n_new_items, axis=0)
                df['y'][-n_new_items:] = y[..., :-self.nb_randomized_params]

            LOGGER.info(f"Test data saved to {test_fp}, \n"
                        f"Shapes: x: {df['x'].shape}, y: {df['y'].shape}")

    def save_data_for_ml_tfrecord(
            self,
            dataset_folder,
            nb_past_timesteps,
            test_fraction=0.2,
            keep_timestamps=False,
            rescale=True,
            positivity=True,
            shuffle=True,
            slice_size=None,
            force_rewrite=False,
    ):
        """
        Write training and test datasets to TFRecord files.
        Original trajectories are optionally scaled and split into training examples:
            (x, y) = (trajectory[i:i+nb_past_timesteps], trajectory[i+nb_past_timesteps])

        Parameters
        ----------
        dataset_folder : folder to save datasets
        nb_past_timesteps : number of steps observed before each transition.
        test_fraction : float, fraction of data that will be used for test.
        keep_timestamps : boolean, whether to keep timestamps in data, default is False.
        rescale : boolean, whether data should be rescaled.
        positivity : boolean, if True, data will be rescaled between 0 and 1, otherwise standardized.
        shuffle : boolean, if True trajectories will be shuffled before producing training examples.
        slice_size : int, number of trajectories to process at once, optional. May be useful for
            large datasets to reduce memory consumption. If None, all trajectories used.
        force_rewrite : boolean, if True, existing files will be rewritten.

        Returns
        -------
        None

        """
        train_gen, test_gen = self.get_train_test_data_generators(
            nb_past_timesteps=nb_past_timesteps,
            test_fraction=test_fraction,
            keep_timestamps=keep_timestamps,
            rescale=rescale,
            positivity=positivity,
            shuffle=shuffle,
            slice_size=slice_size,
        )

        if rescale:
            self._save_scaler(dataset_folder)
            train_fp = os.path.join(dataset_folder, 'train_rescaled.tfrecords')
            test_fp = os.path.join(dataset_folder, 'test_rescaled.tfrecords')
        else:
            train_fp = os.path.join(dataset_folder, 'train.tfrecords')
            test_fp = os.path.join(dataset_folder, 'test.tfrecords')

        if force_rewrite:
            if os.path.exists(train_fp):
                os.remove(train_fp)
            if os.path.exists(test_fp):
                os.remove(test_fp)

        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _create_example(x_arr, y_arr):
            x_shape = x_arr.shape
            x_arr = x_arr.reshape(-1)
            features = tf.train.Features(
                feature={
                    'x': _float_feature(x_arr),
                    'y': _float_feature(y_arr),
                    'x_shape': _int64_feature(x_shape)
                }
            )
            return tf.train.Example(features=features)

        def _process_chunk(x, y):
            for idx in range(x.shape[0]):
                xi = x[idx]
                yi = y[idx]
                example = _create_example(xi, yi)
                writer.write(example.SerializeToString())

        writer = tf.io.TFRecordWriter(train_fp)
        for x, y in train_gen:
            _process_chunk(x, y)

        writer = tf.io.TFRecordWriter(test_fp)
        for x, y in test_gen:
            _process_chunk(x, y)


class BaseDataset(metaclass=abc.ABCMeta):
    """Base class for iterable dataset providing batches of training examples."""

    def __init__(
            self,
            batch_size,
            shuffle=True,
            drop_remainder=True,
    ):
        """
        Initialize Dataset.

        Parameters
        ----------
        batch_size : number of examples in single batch.
        shuffle : boolean, if True, batches will reshuffled each time.
        drop_remainder : boolean, whether to drop the last batch of remaining examples.
            If False, a smaller batch will be yielded at the end.
        """
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_remainder = drop_remainder

    @property
    def batch_size(self):
        return self._batch_size

    @abc.abstractmethod
    def __iter__(self):
        pass


class HDF5Dataset(BaseDataset):
    """Dataset iterating through hdf5 files written by DataTransformer."""

    def __init__(
            self,
            data_file_path,
            batch_size,
            shuffle=True,
            preprocess_fn=None,
    ):
        """
        Initialize Dataset.

        Parameters
        ----------
        data_file_path : path of hdf5 file.
        batch_size : number of examples in single batch.
        shuffle : boolean, if True, batches will reshuffled each time.
        preprocess_fn : callable, function to apply to every data batch before yielding.
        """
        self._data_file_path = data_file_path
        self._preprocess_fn = preprocess_fn

        super().__init__(
            batch_size,
            shuffle=shuffle,
        )

    # def __iter__(self):
    #     """Iterating with effective batch-wise shuffling."""
    #     bs = self.batch_size
    #
    #     with h5py.File(self._data_file_path, 'r', libver='latest') as df:
    #         x = df['x']
    #         y = df['y']
    #         n_examples = x.shape[0]
    #         n_batches = n_examples // bs
    #
    #         batch_idxs = np.arange(0, n_batches)
    #
    #         if self._shuffle:
    #             np.random.shuffle(batch_idxs)
    #
    #         for batch_idx in batch_idxs:
    #             start = batch_idx * bs
    #             end = (batch_idx + 1) * bs
    #             x_batch, y_batch = x[start:end], y[start:end]
    #             if self._preprocess_fn is not None:
    #                 x_batch, y_batch = self._preprocess_fn(x_batch, y_batch)
    #             yield x_batch, y_batch
    #
    #         if self._drop_remainder is False:
    #             x_batch, y_batch = x[n_batches * bs:-1], y[n_batches * bs:-1]
    #             if self._preprocess_fn is not None:
    #                 x_batch, y_batch = self._preprocess_fn(x_batch, y_batch)
    #             yield x_batch, y_batch

    def __iter__(self):
        """Iterating with slower but better example-wise shuffling."""
        bs = self.batch_size

        with h5py.File(self._data_file_path, 'r', libver='latest') as df:
            x = df['x']
            y = df['y']
            n_examples = x.shape[0]
            n_batches = n_examples // bs

            data_idxs = np.arange(0, n_examples)
            batch_idxs = np.arange(0, n_batches)

            if self._shuffle:
                np.random.shuffle(data_idxs)

            for batch_idx in batch_idxs:
                start = batch_idx * bs
                end = (batch_idx + 1) * bs
                idxs = data_idxs[start:end]
                idxs = np.sort(idxs)
                x_batch, y_batch = x[idxs], y[idxs]
                if self._preprocess_fn is not None:
                    x_batch, y_batch = self._preprocess_fn(x_batch, y_batch)
                yield x_batch, y_batch

            if self._drop_remainder is False:
                idxs = data_idxs[n_batches * bs:-1]
                idxs = np.sort(idxs)
                x_batch, y_batch = x[idxs], y[idxs]
                if self._preprocess_fn is not None:
                    x_batch, y_batch = self._preprocess_fn(x_batch, y_batch)
                yield x_batch, y_batch


class TFRecordsDataset(BaseDataset):
    """Dataset iterating through TFRecord files written by DataTransformer."""
    def __init__(
            self,
            records_paths,
            batch_size,
            prefetch_size=None,
            shuffle=True,
            shuffle_buffer_size=100000,
            nb_past_timesteps=None,
            nb_features=None,
            preprocess_fn=None,
    ):
        """
        Initialize Dataset.

        Parameters
        ----------
        records_paths : path to the TFRecord file.
        batch_size : number of examples in single batch.
        shuffle : boolean, if True, batches will reshuffled each time.
        preprocess_fn : callable, function to apply to every data batch before yielding.
        """
        self._records_paths = records_paths
        self._nb_past_timesteps = nb_past_timesteps
        self._nb_features = nb_features
        self._x_len = None
        self._y_len = None

        if (self._nb_past_timesteps is not None) and (self._nb_features is not None):
            self._x_len = self._nb_features * self._nb_past_timesteps
            self._y_len = self._nb_features
            self._parse_record = self._parse_record_known_lengths
        else:
            self._parse_record = self._parse_record_unknown_lengths

        self._prefetch_size = prefetch_size or batch_size * 2  # -1
        self._shuffle_buffer_size = shuffle_buffer_size
        self._preprocess_fn = preprocess_fn

        super().__init__(
            batch_size,
            shuffle=shuffle,
        )

    def __iter__(self):
        config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        graph = tf.Graph()
        dataset = self.create_dataset(graph)
        with tf.compat.v1.Session(graph=graph, config=config) as session:
            dataset_iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
            next_element = dataset_iter.get_next()
            while True:
                try:
                    yield session.run(next_element)
                except tf.errors.OutOfRangeError:
                    break

    @property
    def prefetch_size(self):
        return self._prefetch_size

    def _parse_record_known_lengths(
            self,
            raw_record,
    ):
        features = tf.io.parse_single_example(
            raw_record,
            features={
                'x': tf.io.FixedLenFeature([self._x_len], tf.float32),
                'y': tf.io.FixedLenFeature([self._y_len], tf.float32),
                'x_shape': tf.io.FixedLenFeature([2], tf.int64),
            }
        )
        x = features['x']
        x = tf.reshape(x, features['x_shape'])
        x.set_shape([self._nb_past_timesteps, self._nb_features])
        y = features['y']

        return x, y

    @staticmethod
    def _parse_record_unknown_lengths(
            raw_record,
    ):
        features = tf.io.parse_single_example(
            raw_record,
            features={
                'x': tf.io.VarLenFeature(tf.float32),
                'y': tf.io.VarLenFeature(tf.float32),
                'x_shape': tf.io.FixedLenFeature([2], tf.int64),
            }
        )
        x = tf.sparse.to_dense(features['x'])
        x = tf.reshape(x, features['x_shape'])
        y = tf.sparse.to_dense(features['y'])

        return x, y

    def create_dataset(self, graph=None):
        graph = graph or tf.compat.v1.get_default_graph()
        with graph.as_default():
            dataset = self._create_dataset()
        return dataset

    def _create_dataset(self):
        dataset = tf.data.TFRecordDataset(self._records_paths)
        dataset = dataset.map(self._parse_record)
        if self._preprocess_fn is not None:
            dataset = dataset.map(self._preprocess_fn)
        if self._shuffle is True:
            dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size)
        dataset = dataset.batch(self._batch_size, drop_remainder=self._drop_remainder)
        dataset = dataset.prefetch(self._prefetch_size)
        return dataset
