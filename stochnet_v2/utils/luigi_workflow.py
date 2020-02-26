import logging
import luigi
from luigi.contrib.external_program import ExternalProgramTask
from luigi.util import inherits, requires
from importlib import import_module

from stochnet_v2.utils.file_organisation import ProjectFileExplorer


logger = logging.getLogger('root')


class GlobalParams(luigi.Config):

    model_name = luigi.Parameter()
    project_folder = luigi.Parameter()
    timestep = luigi.FloatParameter()
    dataset_id = luigi.IntParameter()
    endtime = luigi.FloatParameter()
    nb_past_timesteps = luigi.IntParameter()
    params_to_randomize = luigi.Parameter()
    nb_randomized_params = luigi.IntParameter()

    random_seed = luigi.IntParameter()

    # GenerateDataset
    nb_settings = luigi.IntParameter()
    nb_trajectories = luigi.IntParameter()

    # FormatDataset
    positivity = luigi.Parameter()
    test_fraction = luigi.FloatParameter()
    save_format = luigi.Parameter()

    # GenerateHistogramData
    nb_histogram_settings = luigi.IntParameter()
    nb_histogram_trajectories = luigi.IntParameter()
    histogram_endtime = luigi.FloatParameter()

    # Train
    model_id = luigi.IntParameter()
    nb_features = luigi.IntParameter()
    body_config_path = luigi.Parameter()
    mixture_config_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=256)
    add_noise = luigi.Parameter(default='false')
    stddev = luigi.FloatParameter(default=0.01)

    # TrainStatic
    n_epochs = luigi.IntParameter(default=100)

    # TrainSearch
    n_epochs_main = luigi.IntParameter(default=100)
    n_epochs_heat_up = luigi.IntParameter(default=20)
    n_epochs_arch = luigi.IntParameter(default=5)
    n_epochs_interval = luigi.IntParameter(default=5)
    n_epochs_finetune = luigi.IntParameter(default=30)

    # Evaluate
    distance_kind = luigi.Parameter(default='iou')
    target_species_names = luigi.Parameter(default='')
    time_lag_range = luigi.Parameter(default='10')
    settings_idxs_to_save_histograms = luigi.Parameter(default='0')


# @inherits(GlobalParams)
class GenerateDataset(ExternalProgramTask):

    project_folder = GlobalParams().project_folder
    timestep = GlobalParams().timestep
    dataset_id = GlobalParams().dataset_id
    nb_settings = GlobalParams().nb_settings
    nb_trajectories = GlobalParams().nb_trajectories
    endtime = GlobalParams().endtime
    model_name = GlobalParams().model_name
    params_to_randomize = GlobalParams().params_to_randomize
    random_seed = GlobalParams().random_seed

    def program_args(self):
        program_module = import_module("stochnet_v2.scripts.simulate_data_gillespy")
        program_address = program_module.__file__
        return [
            'python',
            program_address,
            f'--project_folder={self.project_folder}',
            f'--timestep={self.timestep}',
            f'--dataset_id={self.dataset_id}',
            f'--nb_settings={self.nb_settings}',
            f'--nb_trajectories={self.nb_trajectories}',
            f'--endtime={self.endtime}',
            f'--model_name={self.model_name}',
            f'--params_to_randomize={self.params_to_randomize}',
            f'--random_seed={self.random_seed}',
        ]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_dataset_file_explorer(self.timestep, self.dataset_id)
        return [
            luigi.LocalTarget(dataset_explorer.settings_fp),
            luigi.LocalTarget(dataset_explorer.dataset_fp),
            luigi.LocalTarget(dataset_explorer.log_fp)
        ]


@requires(GenerateDataset)
class FormatDataset(ExternalProgramTask):

    project_folder = GlobalParams().project_folder
    timestep = GlobalParams().timestep
    dataset_id = GlobalParams().dataset_id
    nb_past_timesteps = GlobalParams().nb_past_timesteps
    nb_randomized_params = GlobalParams().nb_randomized_params
    random_seed = GlobalParams().random_seed

    positivity = GlobalParams().positivity
    test_fraction = GlobalParams().test_fraction
    save_format = GlobalParams().save_format

    def program_args(self):
        program_module = import_module("stochnet_v2.scripts.format_data_for_training")
        program_address = program_module.__file__
        return [
            'python',
            program_address,
            f'--project_folder={self.project_folder}',
            f'--timestep={self.timestep}',
            f'--dataset_id={self.dataset_id}',
            f'--nb_past_timesteps={self.nb_past_timesteps}',
            f'--nb_randomized_params={self.nb_randomized_params}',
            f'--positivity={self.positivity}',
            f'--test_fraction={self.test_fraction}',
            f'--save_format={self.save_format}',
            f'--random_seed={self.random_seed}',
        ]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_dataset_file_explorer(self.timestep, self.dataset_id)
        if self.save_format == 'hdf5':
            cond = 1
        elif self.save_format == 'tfrecord':
            cond = 0
            raise ValueError(f'Save_format {self.save_format} is not supported yet.')
        else:
            raise ValueError(f'save_format parameter not recognized: {self.save_format}, should be `hdf5` or `tfrecord`')
        return [
            luigi.LocalTarget(dataset_explorer.train_fp if cond else dataset_explorer.train_records_fp),
            luigi.LocalTarget(dataset_explorer.test_fp if cond else dataset_explorer.test_records_fp),
            luigi.LocalTarget(dataset_explorer.train_rescaled_fp if cond else dataset_explorer.train_records_rescaled_fp),
            luigi.LocalTarget(dataset_explorer.test_rescaled_fp if cond else dataset_explorer.test_records_rescaled_fp),
            luigi.LocalTarget(dataset_explorer.scaler_fp),
            luigi.LocalTarget(dataset_explorer.log_fp)
        ]


@requires(FormatDataset)
class GenerateHistogramData(ExternalProgramTask):

    project_folder = GlobalParams().project_folder
    timestep = GlobalParams().timestep
    dataset_id = GlobalParams().dataset_id
    model_name = GlobalParams().model_name
    params_to_randomize = GlobalParams().params_to_randomize
    random_seed = GlobalParams().random_seed

    nb_histogram_settings = GlobalParams().nb_histogram_settings
    nb_histogram_trajectories = GlobalParams().nb_histogram_trajectories
    histogram_endtime = GlobalParams().histogram_endtime

    def program_args(self):
        program_module = import_module("stochnet_v2.scripts.simulate_histogram_data_gillespy")
        program_address = program_module.__file__
        return [
            'python',
            program_address,
            f'--project_folder={self.project_folder}',
            f'--timestep={self.timestep}',
            f'--dataset_id={self.dataset_id}',
            f'--nb_settings={self.nb_histogram_settings}',
            f'--nb_trajectories={self.nb_histogram_trajectories}',
            f'--endtime={self.histogram_endtime}',
            f'--model_name={self.model_name}',
            f'--params_to_randomize={self.params_to_randomize}',
            f'--random_seed={self.random_seed}',
        ]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        dataset_explorer = project_explorer.get_dataset_file_explorer(self.timestep, self.dataset_id)
        return [
            luigi.LocalTarget(dataset_explorer.histogram_settings_fp),
            luigi.LocalTarget(dataset_explorer.histogram_dataset_fp),
            luigi.LocalTarget(dataset_explorer.log_fp)
        ]


@requires(FormatDataset)
class TrainStatic(ExternalProgramTask):

    project_folder = GlobalParams().project_folder
    timestep = GlobalParams().timestep
    dataset_id = GlobalParams().dataset_id
    model_id = GlobalParams().model_id
    nb_features = GlobalParams().nb_features
    nb_past_timesteps = GlobalParams().nb_past_timesteps
    nb_randomized_params = GlobalParams().nb_randomized_params
    body_config_path = GlobalParams().body_config_path
    mixture_config_path = GlobalParams().mixture_config_path
    n_epochs = GlobalParams().n_epochs
    batch_size = GlobalParams().batch_size
    add_noise = GlobalParams().add_noise
    stddev = GlobalParams().stddev

    def program_args(self):
        program_module = import_module("stochnet_v2.scripts.train_static")
        program_address = program_module.__file__
        return [
            'python',
            program_address,
            f'--project_folder={self.project_folder}',
            f'--timestep={self.timestep}',
            f'--dataset_id={self.dataset_id}',
            f'--model_id={self.model_id}',
            f'--nb_features={self.nb_features}',
            f'--nb_past_timesteps={self.nb_past_timesteps}',
            f'--nb_randomized_params={self.nb_randomized_params}',
            f'--body_config_path={self.body_config_path}',
            f'--mixture_config_path={self.mixture_config_path}',
            f'--n_epochs={self.n_epochs}',
            f'--batch_size={self.batch_size}',
            f'--add_noise={self.add_noise}',
            f'--stddev={self.stddev}',
        ]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        model_explorer = project_explorer.get_model_file_explorer(self.timestep, self.model_id)
        return [
            luigi.LocalTarget(model_explorer.mixture_config_fp),
            luigi.LocalTarget(model_explorer.body_config_fp),
            luigi.LocalTarget(model_explorer.scaler_fp),
            luigi.LocalTarget(model_explorer.frozen_graph_fp),
            luigi.LocalTarget(model_explorer.graph_keys_fp),
        ]


@requires(FormatDataset)
class TrainSearch(ExternalProgramTask):

    project_folder = GlobalParams().project_folder
    timestep = GlobalParams().timestep
    dataset_id = GlobalParams().dataset_id
    model_id = GlobalParams().model_id
    nb_features = GlobalParams().nb_features
    nb_past_timesteps = GlobalParams().nb_past_timesteps
    nb_randomized_params = GlobalParams().nb_randomized_params
    body_config_path = GlobalParams().body_config_path
    mixture_config_path = GlobalParams().mixture_config_path
    n_epochs_main = GlobalParams().n_epochs_main
    n_epochs_heat_up = GlobalParams().n_epochs_heat_up
    n_epochs_arch = GlobalParams().n_epochs_arch
    n_epochs_interval = GlobalParams().n_epochs_interval
    n_epochs_finetune = GlobalParams().n_epochs_finetune
    batch_size = GlobalParams().batch_size
    add_noise = GlobalParams().add_noise
    stddev = GlobalParams().stddev
    save_format = GlobalParams().save_format

    def program_args(self):
        program_module = import_module("stochnet_v2.scripts.train_search")
        program_address = program_module.__file__
        return [
            'python',
            program_address,
            f'--project_folder={self.project_folder}',
            f'--timestep={self.timestep}',
            f'--dataset_id={self.dataset_id}',
            f'--model_id={self.model_id}',
            f'--nb_features={self.nb_features}',
            f'--nb_past_timesteps={self.nb_past_timesteps}',
            f'--nb_randomized_params={self.nb_randomized_params}',
            f'--body_config_path={self.body_config_path}',
            f'--mixture_config_path={self.mixture_config_path}',
            f'--n_epochs_main={self.n_epochs_main}',
            f'--n_epochs_heat_up={self.n_epochs_heat_up}',
            f'--n_epochs_arch={self.n_epochs_arch}',
            f'--n_epochs_interval={self.n_epochs_interval}',
            f'--n_epochs_finetune={self.n_epochs_finetune}',
            f'--batch_size={self.batch_size}',
            f'--add_noise={self.add_noise}',
            f'--stddev={self.stddev}',
            f'--dataset_kind={self.save_format}',
        ]

    def output(self):
        project_explorer = ProjectFileExplorer(self.project_folder)
        model_explorer = project_explorer.get_model_file_explorer(self.timestep, self.model_id)
        return [
            luigi.LocalTarget(model_explorer.mixture_config_fp),
            luigi.LocalTarget(model_explorer.body_config_fp),
            luigi.LocalTarget(model_explorer.scaler_fp),
            luigi.LocalTarget(model_explorer.frozen_graph_fp),
            luigi.LocalTarget(model_explorer.graph_keys_fp),
        ]


class Evaluate(ExternalProgramTask):

    project_folder = GlobalParams().project_folder
    timestep = GlobalParams().timestep
    dataset_id = GlobalParams().dataset_id
    model_id = GlobalParams().model_id
    model_name = GlobalParams().model_name
    nb_past_timesteps = GlobalParams().nb_past_timesteps
    nb_randomized_params = GlobalParams().nb_randomized_params
    distance_kind = GlobalParams().distance_kind
    target_species_names = GlobalParams().target_species_names
    time_lag_range = GlobalParams().time_lag_range
    settings_idxs_to_save_histograms = GlobalParams().settings_idxs_to_save_histograms

    def requires(self):
        return [
            GenerateHistogramData(),
            TrainSearch()
            # TrainStatic()
        ]

    def program_args(self):
        program_module = import_module("stochnet_v2.scripts.evaluate")
        program_address = program_module.__file__
        return [
            'python',
            program_address,
            f'--project_folder={self.project_folder}',
            f'--timestep={self.timestep}',
            f'--dataset_id={self.dataset_id}',
            f'--model_id={self.model_id}',
            f'--model_name={self.model_name}',
            f'--nb_past_timesteps={self.nb_past_timesteps}',
            f'--nb_randomized_params={self.nb_randomized_params}',
            f'--distance_kind={self.distance_kind}',
            f'--target_species_names={self.target_species_names}',
            f'--time_lag_range={self.time_lag_range}',
            f'--settings_idxs_to_save_histograms={self.settings_idxs_to_save_histograms}',
        ]
