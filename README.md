# StochNetV2

Toolbox for stochastic simulations with CRN models or their deep abstractions. \
Abstract models are based on neural networks predicting a distribution to sample next system state.

## Classes and Modules

### 1. CRN_models.
The module contains base- and example- classes defining CRN models. 
This models can be simulated with Gillepie algorithm provided by `gillespy` package. \
CRN models are used as a source of synthetic data to train and evaluate abstract models.
An instance CRN_model class can
* generate randomized initial concentrations (populations)
* generate randomized reaction rates
* set initial concentrations and reaction rates
* produce trajectories


    Example:
    from stochnet_v2.CRN_models.SIR import SIR
    
    endtime=20
    timestep=0.5
    params_to_randomize = ['beta', 'gamma']
    
    model = SIR(endtime, timestep)
    initial_settings = model.get_initial_settings(n_settings=1)
    randomized_params = model.get_randomized_parameters(params_to_randomize, n_settings=1)
    model.set_species_initial_value(initial_settings)
    model.set_parameters(randomized_params)
    trajectories = model.run()

    
A new CRN model should be inherited from `stochnet_v2.CRN_models.base.BaseCRNModel` 
and have all abstract methods implemented. For examples, see `SIR`, `Bees`, `Gene`, `X16`, or other models.

Some SBML models can be imported with caution: variability of SBML format makes 
automated imports practically infeasible, and for every model some pre-processing is required, e.g. 
editing reaction rates formulas, rewriting reversible reactions as two separate reactions, etc. 
See `BaseSBMLModel` and `EGFR` classes for examples.


### 2. Dataset.
The `dataset` module implements functions and classes to create and operate trajectories data.\
Function `build_simulation_dataset` utilizes multiprocessing tools for faster parallel simulations.\
The `DataTransformer` class prepares generated trajectories by splitting into training examples, 
each being a pair of consecutive system states: (state\[t<sub>i</sub>\], state\[t<sub>i+1</sub>\]).
Resulting dataset is then scaled, split into training and evaluation parts and saved in either TFRecord 
or HDF5 format.\
Classes `HDF5Dataset` and `TFRecordsDataset` can read and iterate through saved datasets, yielding 
data batches. Support shuffling and applying pre-processing functions on the fly.

    endtime = 20
    timestep = 0.5
    params_to_randomize = ['beta', 'gamma']
    dataset_folder = '<dataset-folder-path>'
    dataset_fp = dataset_folder + 'dataset.npy'
    
    dataset = build_simulation_dataset(
        model_name='SIR',
        nb_settings=100,
        nb_trajectories=25,
        timestep=timestep,
        endtime=endtime,
        dataset_folder=dataset_folder,
        params_to_randomize=params_to_randomize
    )
    np.save(dataset_fp, dataset)
    
Returned `dataset` is an array of shape 
(nb_settings * nb_trajectories, n-steps, n-species + n-rand-params.\
One can set the `params_to_randomize` parameter to an empty list to have model 
parameters (i.e. reaction rates) fixed. Then the abstract model will learn to 
predict next state of the system depending on species concentrations only.

    dt = DataTransformer(
        dataset_fp,
        nb_randomized_params=len(params_to_randomize),
    )
    
    dt.save_data_for_ml_hdf5(
        dataset_folder=dataset_folder,
        nb_past_timesteps=1,  # how many states model can see in past, we typically use 1
        test_fraction=0.2,
        keep_timestamps=False,
        rescale=True,
        positivity=False,
        shuffle=True
    )

This will write two files: 'train_rescaled.hdf5' and 'test_rescaled.hdf5', 
each containing 'x' and 'y' columns, where 'x' column corresponds to the system state 
before transition, and 'y' is its next state. 'x' and 'y' will be fed to the neural network 
as input and output correspondingly.

    ds = HDF5Dataset(
        train_rescaled_fp, 
        batch_size=64,
        shuffle=True,
        preprocess_fn=None
    )
    
    for x, y in ds:
        print(x.shape, y.shape)
    
    >out:    
    ((64, 1, 5), (64, 5))  # last dimension is 5 = nb_features + nb_randomized_params
    ((64, 1, 5), (64, 5))
    ...

`Dataset` is python iterable, and yields batches of training examples. 


### 3. StochNet and NASStochNet

#### 3.1.1. StochNet (static model)
`StochNet` class implements an interface for an abstract model. 
It is wrapped around a neural network (Mixture Density Network) which can be trained on 
simulations dataset and then used to produce trajectories.
    
    from stochnet_v2.static_classes.model import StochNet
    
    nb_features = 3  # number of CRN model species
    project_folder = <'project-folder-path'>  # here all files will be stored
    dataset_id = 1  # dataset ID, number
    model_id = 1  # model ID, number
    body_config_path = <'body-config-file-path.json'>
    mixture_config_path = <'mixture-config-file-path.json'>
    
    nn = StochNet(
        nb_past_timesteps=1,
        nb_features=nb_features,
        nb_randomized_params=len(params_to_randomize),
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        body_config_path=body_config_path,
        mixture_config_path=mixture_config_path,
        mode='normal'
    )
    
Body config is a .json file configuring the main (body) part of neural network.
    
    Example of body config:
    
    {
        "body_fn_name": "body_b",
        "block_name": "a",
        "hidden_size": 50,
        "n_blocks": 2,
        "use_batch_norm": false,
        "activation": "relu",
        "activity_regularizer": "none",
        "kernel_constraint": "none",
        "kernel_regularizer": "l2",
        "bias_constraint": "none",
        "bias_regularizer": "l2"
    }

`body_fn_name` and `block_name` parameters represent pre-defined building blocks 
defined in `stochnet_v2.static_classes.nn_bodies`. One can use different combinations 
of `body_fn`, `block`. Parameters `hidden_size` and `n_blocks` define 'width' and 'depth' 
of the network correspondingly. \
Alternatively, one can define custom blocks and body_fn and update `stochnet._get_body_fn` method.\

Mixture config is a .json file configuring the latter (top) part of neural network which represents 
a random distribution (Gaussian Mixture). This distribution is parameterized by the outputs of the 
network body part and then used to sample the next state of the model.
    
    Example of mixture config:
    
    [
	[
		"categorical",
		{
			"hidden_size": "none",      # with a number here, additional layer of this size will be added
			"activation": "none",
			"coeff_regularizer": "none",
			"kernel_constraint": "none",
			"kernel_regularizer": "l2",
			"bias_constraint": "none",
			"bias_regularizer": "l2"
		}
	],
	[
		"normal_diag",
		{
			"hidden_size": "none",      # with a number here, additional layer of this size will be added
			"activation": "none",
			"mu_regularizer": "none",
			"diag_regularizer": "l2",
			"kernel_constraint": "none",
			"kernel_regularizer": "l2",
			"bias_constraint": "none",
			"bias_regularizer": "l2"
		}
	],
	[
		"normal_tril",
		{
			"hidden_size": "none",      # with a number here, additional layer of this size will be added
			"activation": "none",
			"mu_regularizer": "none",
			"diag_regularizer": "l2",
			"sub_diag_regularizer": "l2",
			"kernel_constraint": "none",
			"kernel_regularizer": "l2",
			"bias_constraint": "none",
			"bias_regularizer": "l2"
		}
	],
	
	...
	
	]

Mixture is composed of one 'categorical' random variable and (a combination of) several 'normal' 
components (e.g. 'normal_diag', 'normal_tril', or 'log_normal_diag').


#### 3.1.2. Training (static)

Once StochNet is initialized, it can be trained with Trainer.
    
    from stochnet_v2.static_classes.trainer import Trainer
    from stochnet_v2.static_classes.trainer import ToleranceDropLearningStrategy, ExpDecayLearningStrategy
    
    ckpt_path = Trainer().train(
        nn,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_strategy=learning_strategy,
        ckpt_path=None,  # optional, checkpoint to restore network weights
        dataset_kind=dataset_kind,  # 'hdf5' or 'tfrecord'
        add_noise=True,  # optional data augmentation: add noise to nn inputs
        stddev=0.01,  # standard deviation of added noise
    )

`LearningStrategy` objects define hyper-parameters for training. One can use

    learning_strategy = ToleranceDropLearningStrategy(
        optimizer_type='adam',
        initial_lr=1e-4,
        lr_decay=0.3,
        epochs_tolerance=7,
        minimal_lr=1e-7,
    )
    
which reduces learning rate by factor `lr_decay` every time when train loss didn't 
improve during last `epochs_tolerance` epochs, or

    learning_strategy = ExpDecayLearningStrategy(
        optimizer_type='adam',
        initial_lr=1e-4,
        lr_decay=1e-4,
        lr_cos_steps=5000,
        lr_cos_phase=np.pi / 2,
        minimal_lr=1e-7,
    )
    
which updates learning rate every training step so that it decays exponentially. 
If `lr_cos_steps` is non-zero, it will have a 'saw'-like shape (with exponentially decaying pikes).

Once training is finished, all necessary files are saved to the model folder 
(see sect. File Organisation). After this, the model can be loaded in 'inference' 
mode for simulations any time.


#### 3.2.1. NASStochNet (dynamic, Neural Architecture Search)
As an alternative to custom selection, we can search for the configuration of the body-part 
of StochNet. An over-parameterized, brunching network is built first, where instead of one 
concrete layer it has a set of layer-candidates, and these mix-layers are all inter-connected. 
Then the two-level optimisation is performed by `stochnet_v2.dynamic_classes.trainer import Trainer` 
to find the best layer-candidates and connections between them. \

    from stochnet_v2.dynamic_classes.model import NASStochNet
    from stochnet_v2.dynamic_classes.trainer import Trainer
    
    nn = NASStochNet(
        nb_past_timesteps=1,
        nb_features=nb_features,
        nb_randomized_params=len(params_to_randomize),
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        body_config_path=body_config_path,
        mixture_config_path=mixture_config_path,
        mode='normal'
    )

Model config is different in this case:

    {
        "n_cells": 2,
        "cell_size": 2,
        "expansion_multiplier": 30,
        "n_states_reduce": 2,
        "kernel_constraint": "none",
        "bias_constraint": "none",
        "kernel_regularizer": "l2",
        "bias_regularizer": "l2",
        "activity_regularizer": "none"
    }
    
The network body is constructed of `n_cells` cells, each having `cell_size` mix-layers.
Every mix-layer represents a set of candidate operations, listed in 
`stochnet_v2.dynamic_classes.genotypes.PRIMITIVES`. The latter `n_states_reduce` states 
of a cell are averaged to produce its output. `expansion_multiplier` parameter controls 
the 'width' of the network: if an expanding cell receives `n` features, it will output 
`n * expansion_multiplier` features. See https://arxiv.org/abs/2002.01889 for further details.\

Mixture config is the same as in section 3.1.1.


#### 3.2.2. Training (Architecture Search)

For the Architecture Search, training consists of two stages: 
 * searching for optimal configuration
 * fine-tuning of the found architecture after all redundancies are pruned
 
The main (search) stage is a two-level optimisation, i.e. we run two separate optimisation 
procedures in altering manner for several epochs each:

 * (main): update network parameters (weights in layers) for `n_epochs_interval` epochs
 * (arch): update architecture parameters (weights of candidate operations in mix-layer) 
 for `n_epochs_arch` epochs

Optionally, we let the layers adjust to data for the first few (`n_epochs_heat_up`) epochs.

    ckpt_path = Trainer().train(
        nn,
        n_epochs_main=100,  # total number of epochs in (main) optimisation procedure
        n_epochs_heat_up=10,
        n_epochs_arch=3,
        n_epochs_interval=5,
        n_epochs_finetune=20,
        batch_size=batch_size,
        learning_strategy_main=learning_strategy_main,
        learning_strategy_arch=learning_strategy_arch,
        learning_strategy_finetune=learning_strategy_finetune,
        ckpt_path=None,
        dataset_kind='hdf5',
        add_noise=True,
        stddev=0.01,
        mode=['search', 'finetune']
    )
    
    learning_strategy_main = ToleranceDropLearningStrategy(
        optimizer_type='adam',
        initial_lr=1e-4,
        lr_decay=0.3,
        epochs_tolerance=7,
        minimal_lr=1e-7,
    )
    
    learning_strategy_arch = ToleranceDropLearningStrategy(
        optimizer_type='adam',
        initial_lr=1e-3,
        lr_decay=0.5,
        epochs_tolerance=20,
        minimal_lr=1e-7,
    )
    
    learning_strategy_finetune = ToleranceDropLearningStrategy(
        optimizer_type='adam',
        initial_lr=1e-4,
        lr_decay=0.3,
        epochs_tolerance=5,
        minimal_lr=1e-7,
    )

After the search and fine-tuning stages, all necessary files are saved to the model folder, 
and the model can be loaded in 'inference' mode for simulations. Either `StochNet` or `NASStochNet` 
can be used to load trained model and run simulations.


#### 3.3. Simulations with StochNet

For simulations, trained model can be loaded in 'inference' mode:

    nn = StochNet(
        nb_past_timesteps=1,
        nb_features=nb_features,
        nb_randomized_params=len(params_to_randomize),
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        mode='inference'
    )

Get random inputs (initial state and randomized parameters) from CRN model:
    
    from stochnet_v2.CRN_models.SIR import SIR
    from stochnet_v2.CRN_models.utils.util import merge_species_and_param_settings
    
    n_settings = 100
    model = SIR(endtime, timestep)
    initial_settings = model.get_initial_settings(n_settings)
    randomized_params = model.get_randomized_parameters(params_to_randomize, n_settings)
    inputs = merge_species_and_param_settings(initial_settings, randomized_params)
    
StochNet model takes arrays of shape 
(n_settings, nb_past_timesteps, nb_features + nb_randomized_params) for inputs. 
If `params_to_randomize` was an empty list originally, only `initial_settings` should be used 
as `inputs`.

    next_state_values = nn.next_state(
        curr_state_values=inputs[:, np.newaxis, :],
        curr_state_rescaled=False,
        scale_back_result=True,
        round_result=False,
        n_samples=100,
    )
    
This will produce an array of shape (n_samples, n_settings, 1, nb_features) containing predictions 
for the next state of the system, i.e. its state after `timestep` time.

To simulate trajectories starting from `inputs`, run

    nn_traces = nn.generate_traces(
        curr_state_values=inputs[:, np.newaxis, :],
        n_steps=n_steps,
        n_traces=traj_per_setting,
        curr_state_rescaled=False,
        scale_back_result=True,
        round_result=True,
        add_timestamps=True
    )

Returned array of trajectories has shape (n_settings, n_traces, n_steps, nb_features).


#### 3.4. Evaluation

To evaluate the quality of abstract models, we compare distributions (histograms) of species 
of interest. For this, we simulate many trajectories of the original model starting from 
a set of random initial settings:
    
    nb_histogram_settings = 25
    nb_histogram_trajectories = 2000
    histogram_settings_fp = <'histogram-settings-file-path'>
    
    histogram_settings = crn_class.get_initial_settings(nb_settings)
    np.save(histogram_settings_fp, histogram_settings)
    
    histogram_dataset = build_simulation_dataset(
        model_name,
        nb_histogram_settings,
        nb_histogram_trajectories,
        timestep,
        endtime,
        dataset_folder,
        params_to_randomize=params_to_randomize,
        prefix='histogram_partial_',
        how='stack',
        settings_filename=os.path.basename(histogram_settings_fp),
    )
    np.save(dataset_explorer.histogram_dataset_fp, histogram_dataset)

Then evaluation script runs StochNet the simulations starting from the same initial settings 
as the trajectories in `histogram_dataset`. \
Evaluation script saves:
* overall average value of histogram distance
* histogram dataset self-distance (as a lower bound for the distance between models)
* plots of species histograms after different number of steps
* plots of average (over different settings) distance between histograms produced by original and abstract 
model after different number of time-steps.

Example:


    from stochnet_v2.utils.evaluation import evaluate
    
    distance_kind = 'dist'
    target_species_names = ['S', 'I']
    time_lag_range = [1, 5, 10, 20]
    settings_idxs_to_save_histograms = [i for i in range(10)]
    
    evaluate(
        model_name=model_name,
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        nb_randomized_params=len(params_to_randomize),
        nb_past_timesteps=1,
        n_bins=100,
        distance_kind=distance_kind,
        with_timestamps=True,
        save_histograms=True,
        time_lag_range=time_lag_range,
        target_species_names=target_species_names,
        path_to_save_nn_traces=nn_histogram_data_fp,
        settings_idxs_to_save_histograms=settings_idxs_to_save_histograms,
    )


### 4. Utils

#### 4.1. Helper functions

Various helper functions are implemented in `stochnet_v2.utils`:
 * `benchmarking`: measure simulation times for original CRN models (Gillespie algorithm) 
 and MDN-based abstract models.
 * `evaluation`: generate trajectories for histograms, measure histogram distance, plot histograms
 * `util`: 
    * `generate_gillespy_traces` - generate trajectories for original CRN model using multiprocessing tools
    * `plot_traces`, `plot_random_traces` - plot model trajectories
    * `visualize_description` - visualize parameters of the components of mixture distribution
    * `visualize_genotypes` - visualize NAS model 'genotype', i.e. the architecture selected by the architecture search algorithm:
    selected operation candidates and connections

    
            # generate traces:
            
            initial_settings = model.get_initial_settings(n_settings)
            
            gillespy_traces = generate_gillespy_traces(
                settings=initial_settings,
                step_to=n_steps,
                timestep=timestep,
                gillespy_model=m,
                traj_per_setting=traj_per_setting,
            )
            
            nn_traces = nn.generate_traces(
                initial_settings[:, np.newaxis, :],
                n_steps=n_steps,
                n_traces=traj_per_setting,
                curr_state_rescaled=False,
                scale_back_result=True,
                round_result=True,
                add_timestamps=True
            )
            
            # plot traces:
            
            setting_idx = 0
            n_traces = 5
            
            plt.figure(figsize=(16, 10))
            plot_random_traces(gillespy_traces[setting_idx], n_traces, linestyle='--', marker='')
            plot_random_traces(nn_traces[setting_idx], n_traces, linestyle='-', marker='')
            
            # visualize distribution parameters:
            
            curr_state = np.expand_dims(initial_settings[setting_idx:setting_idx + 1], -2)
            descr = nn.get_description(
                current_state_val=curr_state, 
                current_state_rescaled=False, 
                visualize=False
            )
            figures = visualize_description(descr, save_figs_to='./visualizations')
            
            # visualize NAS model:
            
            # genotypes.pickle is automatically saved at the end of architecture search (sect. 3.2.2) 
            # in the model folder. 
        
            with open(os.path.join(nn.model_explorer.model_folder, 'genotypes.pickle'), 'rb') as f:
                genotypes = pickle.load(f)
            
            visualize_genotypes(genotypes, './visualizations/model_genotype.pdf')
            

### 4.2. File Organisation

We use `FileExplorer` helper classes to maintain uniform file structure. 
These classes store paths for saving and reading files, such as dataset files, model files, configs, 
evaluation results, etc. See `stochnet_v2.utils.file_organisation` for details.

    model_name = 'SIR'
    timestep = 0.5
    dataset_id = 1
    model_id = 1
    nb_features = 4
    nb_past_timesteps = 1
    params_to_randomize = ['beta', 'gamma']
    
    project_folder = '~/DATA/' + model_name
    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)
    model_explorer = project_explorer.get_model_file_explorer(timestep, model_id)
    histogram_explorer = dataset_explorer.get_histogram_file_explorer(model_id, 0)
    
`ProjectFileExplorer` creates root folders for models and data:
    
    * SIR/
        * dataset/
        * models/

`DatasetFileExplorer` creates separate folders for different datasets:

    * dataset/
        * data/
            * 0.5/        # timestep
                * 1/      # dataset_id

`ModelFileExplorer` creates separate folders for different models:

    * models/
        * 0.5/            # timestep
            * 1/          # model_id
    
`HistogramFileExplorer` creates folders to store histograms during evaluation:

    * * dataset/
        * data/
            * 0.5/
                * 1/
                    * histogram/
                        * model_<model-id>/
                            * setting_<setting-idx>/
                                * <time-lag>/


### 5. High-level scripts

After properly defining a CRN model, the workflow takes the following actions:
* generate dataset of trajectories
* generate histogram-dataset (another dataset for evaluation)
* format dataset (transform trajectories to training examples)
* train model (static or NAS)
* evaluate

Module `stochnet_v2.scripts` contains scripts to run these tasks:

    python stochnet_v2/scripts/simulate_data_gillespy.py \
           --project_folder='home/DATA/SIR' \
           --timestep=0.5 \
           --dataset_id=1 \
           --nb_settings=100 \
           --nb_trajectories=50 \
           --endtime=20 \
           --model_name='SIR' \
           --params_to_randomize='beta gamma'

    python stochnet_v2/scripts/simulate_histogram_data_gillespy.py \
           --project_folder='/home/DATA/SIR' \
           --timestep=0.5 \
           --dataset_id=1 \
           --nb_settings=10 \
           --nb_trajectories=2000 \
           --endtime=20 \
           --model_name='SIR' \
           --params_to_randomize='beta gamma'

    python stochnet_v2/scripts/format_data_for_training.py \
           --project_folder='/home/DATA/SIR' \
           --timestep=0.5 \
           --dataset_id=1 \
           --nb_past_timesteps=1 \
           --nb_randomized_params=2 \
           --positivity=true \
           --test_fraction=0.2 \
           --save_format='hdf5'

    python stochnet_v2/scripts/train_static.py \
        --project_folder='/home/DATA/SIR' \
        --timestep=0.5 \
        --dataset_id=1 \
        --model_id=1 \
        --nb_features=3 \
        --nb_past_timesteps=1 \
        --nb_randomized_params=2 \
        --body_config_path='/home/DATA/SIR/body_config.json' \
        --mixture_config_path='/home/DATA/SIR/mixture_config.json' \
        --n_epochs=100 \
        --batch_size=256 \
        --add_noise=True \
        --stddev=0.01

    python stochnet_v2/scripts/train_search.py \
        --project_folder='/home/DATA/SIR' \
        --timestep=0.5 \
        --dataset_id=1 \
        --model_id=2 \
        --nb_features=3 \
        --nb_past_timesteps=1 \
        --nb_randomized_params=2 \
        --body_config_path='/home/DATA/SIR/body_config_search.json' \
        --mixture_config_path='/home/DATA/SIR/mixture_config_search.json' \
        --n_epochs_main=100 \
        --n_epochs_heat_up=10 \
        --n_epochs_arch=5 \
        --n_epochs_interval=5 \
        --n_epochs_finetune=20 \
        --batch_size=256 \
        --add_noise=True \
        --stddev=0.01

    python stochnet_v2/scripts/evaluate.py \
        --project_folder='/home/DATA/SIR' \
        --timestep=0.5 \
        --dataset_id=1 \
        --model_id=2 \
        --model_name='SIR' \
        --nb_past_timesteps=1 \
        --nb_randomized_params=2 \
        --distance_kind='dist' \
        --target_species_names='S I' \
        --time_lag_range='1 3 5 10 20' \
        --settings_idxs_to_save_histograms='0 1 2 3 4 5 6 7 8 9'

Config files for MDN body and mixture parts should be filled as shown in Sect. 3.


### 6. Luigi workflow manager

The above workflow is wrapped with `luigi` library designed for running complex pipelines 
of inter-dependent tasks. \
Alternatively to manual running above commands, one can fill a luigi configuration file, 
and it will run the whole sequence of tasks taking care of the right order and pre-requisites 
for every task. \
Example of luigi config file `SIR.cfg`:

    [GlobalParams]
    
    model_name=SIR
    project_folder=/home/DATA/SIR
    timestep=0.5
    endtime=20
    dataset_id=1
    nb_past_timesteps=1
    params_to_randomize=beta gamma
    nb_randomized_params=2
    random_seed=42
    
    nb_settings=100
    nb_trajectories=50
    
    positivity=true
    test_fraction=0.2
    save_format=hdf5
    
    nb_histogram_settings=25
    nb_histogram_trajectories=2000
    histogram_endtime=20
    
    model_id=2
    nb_features=3
    body_config_path=/home/DATA/SIR/body_config_search.json
    mixture_config_path=/home/DATA/SIR/mixture_config_search.json
    n_epochs=100
    n_epochs_main=100
    n_epochs_heat_up=10
    n_epochs_arch=5
    n_epochs_interval=5
    n_epochs_finetune=20
    batch_size=256
    add_noise=true
    stddev=0.01
    
    distance_kind=dist
    target_species_names=S I
    time_lag_range=1 3 5 10 20
    settings_idxs_to_save_histograms=0 1 2 3 4 5 6 7 8 9

Then to run all tasks from data generation to evaluation run

    LUIGI_CONFIG_PATH=/home/DATA/SIR/SIR.cfg python -m luigi --module stochnet_v2.utils.luigi_workflow Evaluate --log-level=INFO --local-scheduler

One can also run these tasks one by one by replacing `Evaluate` task name with 
`GenerateDataset`, `FormatDataset`, `GenerateHistogramData`, `TrainStatic` or `TrainSearch`.

> *Note*: At the moment, to switch dependencies of `Evaluate` task between 
TrainStatic and TrainSearch, one should leave uncommented corresponding line in 
`stochnet_v2.utils.luigi_workflow`, in definition of `Evaluate` class:

    def requires(self):
        return [
            GenerateHistogramData(),
            TrainSearch()
            # TrainStatic()
        ]


### 7. GridRunner

`GridRunner` implements simulation of multiple CRN instances on a (spatial) grid 
with communication via spreading a subset of species across neighboring grid nodes. \
`GridRunner` is initialized with a model and `GridSpec`.
`GridSpec` specifies a grid:

    from stochnet_v2.static_classes.grid_runner import GridSpec
    
    grid_spec = GridSpec(
        boundaries=[[0.0, 1.0], [0.0, 1.0]],
        grid_size=[10, 10]
    )

A model can be either an instance of trained `StochNet`, or an instance of special proxy-class for 
`CRN_model`:


    from stochnet_v2.CRN_models.Bees import Bees
    from stochnet_v2.static_classes.grid_runner import Model
    
    timestep = 0.5
    endtime = 100.0
    params_to_randomize = []
    
    m = Bees(endtime, timestep)
    model = Model(m, params_to_randomize=[])

or
    
    from stochnet_v2.static_classes.model import StochNet
    
    model_id = 1
    dataset_id = 1
    nb_features = 4
    nb_past_timesteps = 1
    
    model = StochNet(
        nb_past_timesteps=nb_past_timesteps,
        nb_features=nb_features,
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        nb_randomized_params=len(params_to_randomize),
        mode='inference'
    )

`GridRunner.state` stores state values for every model instance. 
This state can be updated by running one of the following steps:
* `model_step`, which for every grid node runs one forward step of the model, 
   starting corresponding state values stored in `GridRunner.state`.
* `diffusion_step`, which simulates diffusion of species across the grid with a 
   Gaussian diffusion kernel. A subset of species can be selected for this step.
* `max_propagation_step`, which assigns to every grid node the (factored) maximum value 
   of its neighbors. A subset of species can be selected for this step.

Example:


    from stochnet_v2.static_classes.grid_runner import GridRunner
    
    gr = GridRunner(
        model,
        grid_spec,
        save_dir='/home/DATA/GridRunner/Bees',
        diffusion_kernel_size=3,
        diffusion_sigma=0.7
    )
    
    gr.model_step()
    
    gr.diffusion_step(
        species_idx=3,
        conservation=False,
    )
    
    gr.max_propagation_step(
        species_idx=3,
        alpha=0.5,
    )


`Bees` model describes the defense behavior of honeybees: after stinging, 
a bee dies and releases pheromone (species_idx=3). In presence of pheromone 
other bees become aggressive and can sting too. \
To model this behavior, we first spread groups of bees across a grid, and set initial 
concentrations of pheromone in several places (or alternatively making some of them 
initially aggressive). 
Then by altering model steps and pheromone propagation steps we obtain a discretized approximation 
of the colony behavior.
 
     states_sequence = gr.run_model(
        tot_iterations=100,      # total number of iterations
        propagation_steps=1,     # every iteration has 1 propagation step
        model_steps=5,           # every iteration has 5 model step
        propagation_mode='mp',   # 'mp' for max_propagation_step, 'd' for diffusion_step
        species_idx=3,           # keyword-arguments for propagation steps
        alpha=0.33               # keyword-arguments for propagation steps
    )

`states_sequence` can then be animated:

![alt text](https://github.com/dennerepin/gif/blob/master/animated_progress.gif?raw=true)


\
\
Original idea of using Mixture Density Networks was proposed by Luca Bortolussi & Luca Palmieri 
(https://link.springer.com/chapter/10.1007/978-3-319-99429-1_2).
Some code was taken from: https://github.com/LukeMathWalker/StochNet.
