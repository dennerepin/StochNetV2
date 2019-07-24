import abc
import h5py
import numpy as np
import os
import pickle
import tensorflow as tf

from bidict import bidict
from sklearn.preprocessing.data import StandardScaler, MinMaxScaler
from tqdm import tqdm
from stochnet_v2.utils.errors import ShapeError


class DataTransformer:

    def __init__(
            self,
            dataset_address,
            with_timestamps=True,
            dtype=np.float32,
    ):
        self.nb_trajectories = None
        self.nb_timesteps = None
        self.nb_features = None
        self.labels = None
        self.with_labels = False
        self.with_timestamps = with_timestamps

        self._scaler = None
        self.scaler_is_fitted = False
        self.scaler_positivity = None
        self.rescaled = False
        self.dtype = dtype

        self.read_data(dataset_address)

    @property
    def scaler(self):
        return self._scaler

    def read_data(self, dataset_address):
        with open(dataset_address, 'rb') as data_file:
            self.data = np.asarray(np.load(data_file), dtype=self.dtype)
            self._memorize_dataset_shape()

    def _memorize_dataset_shape(self):
        if self.data.ndim != 3:
            raise ShapeError(
                f"The dataset is not properly formatted.\n"
                f"We expect the following shape: "
                f"(nb_trajectories, nb_timesteps, nb_features),\n"
                f"got: {self.data.shape}"
            )
        self.nb_trajectories, self.nb_timesteps, self.nb_features = self.data.shape

    def set_labels(self, labels):
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

    def rescale(self, positivity=True, slice_size=None):
        if self.rescaled is False:
            if self.scaler_is_fitted is False:
                self._fit_scaler(positivity, slice_size)
            self._apply_scaler(slice_size)

        elif self.scaler_positivity != positivity:
            if self.rescaled:
                self.scale_back(slice_size)
            self._create_scaler(positivity)
            self._fit_scaler(positivity, slice_size)
            self._apply_scaler(slice_size)

    def _create_scaler(self, positivity):
        self.scaler_positivity = positivity
        if positivity is True:
            eps = 1e-9
            self._scaler = MinMaxScaler(feature_range=(eps, 1))
        else:
            self._scaler = StandardScaler()

    def _fit_scaler(self, positivity=True, slice_size=None):
        if self._scaler is None:
            self._create_scaler(positivity)

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

    def _apply_scaler(self, slice_size=None):
        if slice_size is None:
            flat_data = self.data.reshape(-1, self.nb_features)
            flat_data = self._scaler.transform(flat_data)
            self.data = flat_data.reshape(self.nb_trajectories, self.nb_timesteps, self.nb_features)
        else:
            n_slices = self.nb_trajectories // slice_size

            for i in tqdm(range(n_slices)):
                data_slice = self.data[i * slice_size: (i + 1) * slice_size, ...]
                data_slice = data_slice.reshape(-1, self.nb_features)
                data_slice = self._scaler.transform(data_slice)
                self.data[i * slice_size: (i + 1) * slice_size, ...] = \
                    data_slice.reshape(-1, self.nb_timesteps, self.nb_features)

            if self.nb_trajectories % slice_size != 0:
                data_slice = self.data[n_slices * slice_size:, ...]
                data_slice = data_slice.reshape(-1, self.nb_features)
                data_slice = self._scaler.transform(data_slice)
                self.data[n_slices * slice_size:, ...] = \
                    data_slice.reshape(-1, self.nb_timesteps, self.nb_features)

        self.rescaled = True

    def scale_back(self, slice_size=None):
        if self.rescaled:
            if slice_size is None:
                flat_data = self.data.reshape(-1, self.nb_features)
                flat_data = self._scaler.inverse_transform(flat_data)
                self.data = flat_data.reshape(self.nb_trajectories, self.nb_timesteps, self.nb_features)
            else:
                n_slices = self.nb_trajectories // slice_size

                for i in tqdm(range(n_slices)):
                    data_slice = self.data[i * slice_size: (i + 1) * slice_size, ...]
                    data_slice = data_slice.reshape(-1, self.nb_features)
                    data_slice = self._scaler.inverse_transform(data_slice)
                    self.data[i * slice_size: (i + 1) * slice_size, ...] = \
                        data_slice.reshape(-1, self.nb_timesteps, self.nb_features)

                if self.nb_trajectories % slice_size != 0:
                    data_slice = self.data[n_slices * slice_size:, ...]
                    data_slice = data_slice.reshape(-1, self.nb_features)
                    data_slice = self._scaler.inverse_transform(data_slice)
                    self.data[n_slices * slice_size:, ...] = \
                        data_slice.reshape(-1, self.nb_timesteps, self.nb_features)

            self.rescaled = False

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
            yield x_data, y_data

    def _train_test_generators(
            self,
            nb_past_timesteps,
            test_fraction=0.2,
            slice_size=None,
    ):
        n_train_trajectories = int((1. - test_fraction) * self.nb_trajectories)

        train_gen = self._transitions_generator(
            self.data[:n_train_trajectories],
            nb_past_timesteps,
            slice_size
        )
        test_gen = self._transitions_generator(
            self.data[n_train_trajectories:],
            nb_past_timesteps,
            slice_size
        )
        return train_gen, test_gen

    def get_train_test_data_generators(
            self,
            nb_past_timesteps,
            test_fraction=0.2,
            keep_timestamps=False,
            rescale=True,
            positivity=True,
            shuffle=True,
            slice_size=None,

    ):
        if keep_timestamps is False:
            self.drop_timestamps()

        if rescale is True:
            self.rescale(positivity, slice_size)
        elif rescale is False and self.rescaled is True:
            self.scale_back(slice_size)

        if shuffle is True:
            self._shuffle_data()

        return self._train_test_generators(
            nb_past_timesteps,
            test_fraction,
            slice_size,
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
            nb_past_timesteps,
            test_fraction=0.2,
            keep_timestamps=False,
            rescale=True,
            positivity=True,
            shuffle=True,
            slice_size=None,
            force_rewrite=False,
    ):
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
                shape=(0, self.nb_features),
                maxshape=(None, self.nb_features),
                chunks=True,
            )
            for x, y in train_gen:
                n_new_items = x.shape[0]
                df['x'].resize(df['x'].shape[0] + n_new_items, axis=0)
                df['x'][-n_new_items:] = x
                
                df['y'].resize(df['y'].shape[0] + n_new_items, axis=0)
                df['y'][-n_new_items:] = y

            print(f"Train data saved to {train_fp}, \n"
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
                shape=(0, self.nb_features),
                maxshape=(None, self.nb_features),
                chunks=True,
            )
            for x, y in test_gen:
                n_new_items = x.shape[0]
                df['x'].resize(df['x'].shape[0] + n_new_items, axis=0)
                df['x'][-n_new_items:] = x

                df['y'].resize(df['y'].shape[0] + n_new_items, axis=0)
                df['y'][-n_new_items:] = y

            print(f"Test data saved to {test_fp}, \n"
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
    ):
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

    def __init__(
            self,
            batch_size,
            prefetch_size=None,
            shuffle=False,
            shuffle_buffer_size=100,
    ):
        self._batch_size = batch_size
        self._prefetch_size = prefetch_size or -1

        self._shuffle = shuffle
        self._shuffle_buffer_size = shuffle_buffer_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def prefetch_size(self):
        return self._prefetch_size

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

    @abc.abstractmethod
    def _create_dataset(self):
        pass

    def create_dataset(self, graph=None):
        if graph is not None:
            with graph.as_default():
                dataset = self._create_dataset()
                if self._shuffle is True:
                    dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size)
                dataset = dataset.batch(self._batch_size, drop_remainder=True)
                dataset = dataset.prefetch(self._prefetch_size)
        else:
            dataset = self._create_dataset()
            if self._shuffle is True:
                dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size)
            dataset = dataset.batch(self._batch_size, drop_remainder=True)
            dataset = dataset.prefetch(self._prefetch_size)

        return dataset


class TFRecordsDataset(BaseDataset):

    def __init__(
            self,
            records_paths,
            batch_size,
            prefetch_size=None,
            shuffle=False,
            shuffle_buffer_size=100,
            nb_past_timesteps=None,
            nb_features=None,
    ):
        self._records_paths = records_paths
        self._nb_past_timesteps= nb_past_timesteps
        self._nb_features = nb_features
        self._x_len = None
        self._y_len = None

        if (self._nb_past_timesteps is not None) and (self._nb_features is not None):
            self._x_len = self._nb_features * self._nb_past_timesteps
            self._y_len = self._nb_features
            self._parse_record = self._parse_record_known_lengths
        else:
            self._parse_record = self._parse_record_unknown_lengths

        super().__init__(
            batch_size,
            prefetch_size=prefetch_size,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
        )

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
        y = features['y']

        return x, y

    @staticmethod
    def _parse_record_unknown_lengths(
            raw_record,
    ):
        features = tf.parse_single_example(
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

    def _create_dataset(self):
        dataset = tf.data.TFRecordDataset(self._records_paths)
        dataset = dataset.map(self._parse_record)
        return dataset
