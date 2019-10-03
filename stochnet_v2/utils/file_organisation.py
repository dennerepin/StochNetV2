import os


def maybe_create_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


class ProjectFileExplorer:

    def __init__(
            self,
            project_folder,
    ):
        self.project_folder = project_folder

        self.data_root_folder = os.path.join(project_folder, 'dataset/data')
        maybe_create_dir(self.data_root_folder)

        self.models_root_folder = os.path.join(project_folder, 'models')
        maybe_create_dir(self.models_root_folder)

    def get_dataset_file_explorer(self, timestep, dataset_id):
        return DatasetFileExplorer(self.data_root_folder, timestep, dataset_id)

    def get_model_file_explorer(self, timestep, model_id):
        return ModelFileExplorer(self.models_root_folder, timestep, model_id)


class DatasetFileExplorer:

    def __init__(
            self,
            data_root_folder,
            timestep,
            dataset_id,
    ):
        self.data_root_folder = data_root_folder
        self.dataset_id = dataset_id
        self.timestep = timestep
        self.dataset_folder = os.path.join(
            self.data_root_folder,
            str(self.timestep),
            str(self.dataset_id),
        )
        maybe_create_dir(self.dataset_folder)

        self.settings_fp = os.path.join(self.dataset_folder, 'settings.npy')
        self.log_fp = os.path.join(self.dataset_folder, 'log.txt')
        self.dataset_fp = os.path.join(self.dataset_folder, 'dataset.npy')

        self.train_fp = os.path.join(self.dataset_folder, 'train.hdf5')
        self.test_fp = os.path.join(self.dataset_folder, 'test.hdf5')
        self.train_rescaled_fp = os.path.join(self.dataset_folder, 'train_rescaled.hdf5')
        self.test_rescaled_fp = os.path.join(self.dataset_folder, 'test_rescaled.hdf5')

        self.train_records_fp = os.path.join(self.dataset_folder, 'train.tfrecords')
        self.test_records_fp = os.path.join(self.dataset_folder, 'test.tfrecords')
        self.train_records_rescaled_fp = os.path.join(self.dataset_folder, 'train_rescaled.tfrecords')
        self.test_records_rescaled_fp = os.path.join(self.dataset_folder, 'test_rescaled.tfrecords')

        self.scaler_fp = os.path.join(self.dataset_folder, 'scaler.pickle')
        self.histogram_settings_fp = os.path.join(self.dataset_folder, 'histogram_settings.npy')
        self.histogram_dataset_fp = os.path.join(self.dataset_folder, 'histogram_dataset.npy')

    def get_histogram_file_explorer(self, model_id, nb_steps):
        return HistogramFileExplorer(self.dataset_folder, model_id, nb_steps)


class ModelFileExplorer:

    def __init__(
            self,
            models_root_folder,
            timestep,
            model_id,
    ):
        self.models_root_folder = models_root_folder
        self.model_id = model_id
        self.timestep = timestep
        self.model_folder = os.path.join(
            self.models_root_folder,
            str(self.timestep),
            str(self.model_id),
        )
        maybe_create_dir(self.model_folder)
        self.log_fp = os.path.join(self.model_folder, 'log.txt')
        self.frozen_graph_fp = os.path.join(self.model_folder, 'frozen_graph.pb')
        self.graph_keys_fp = os.path.join(self.model_folder, 'graph_keys.json')
        self.mixture_config_fp = os.path.join(self.model_folder, 'mixture_config.json')
        self.body_config_fp = os.path.join(self.model_folder, 'body_config.json')
        self.scaler_fp = os.path.join(self.model_folder, 'scaler.pickle')


class HistogramFileExplorer:

    def __init__(
            self,
            dataset_folder,
            model_id,
            nb_steps,
    ):
        self.dataset_folder = dataset_folder
        self.model_id = model_id
        self.model_histogram_folder = os.path.join(
            self.dataset_folder,
            f'histogram/model_{self.model_id}',
        )
        self.histogram_folder = os.path.join(
            self.model_histogram_folder,
            str(nb_steps),
        )
        maybe_create_dir(self.histogram_folder)
        self.log_fp = os.path.join(self.histogram_folder, 'log.txt')

