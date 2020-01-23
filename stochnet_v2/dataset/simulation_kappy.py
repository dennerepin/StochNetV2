import kappy
import logging
import multiprocessing
import numpy as np
import os
import pickle
import re
from functools import partial

from stochnet_v2.dataset.simulation_gillespy import _concatenate_simulations
from stochnet_v2.dataset.simulation_gillespy import _stack_simulations
from stochnet_v2.dataset.simulation_gillespy import _save_simulation_data

LOGGER = logging.getLogger('scripts.kappa_simulation')


def build_simulation_dataset(
        model_name,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_folder,
        var_list,
        prefix='partial_',
        how='concat',
        settings_filename='settings.pickle',
):
    """
        Produce dataset of simulations.
        Runs Kappa simulations of selected model w.r.t given
        parameters (discrete time-step, number of initial settings,
        and number of simulations for each of the initial settings).
        Initial settings are stored in dataset_folder with name settings_filename.
        Depending on `how` parameter, simulated trajectories are gathered
        in particularly shaped multi-dimensional array.

        Parameters
        ----------
        model_name : path to .ka Kappa model file.
        nb_settings : number of initial states for simulations.
            Initial states are (randomly) produced by `get_random_initial_settings`.
        nb_trajectories : number of trajectories to simulate starting from each of the initial states.
        timestep : discrete time-step for simulations.
        endtime : end-time for simulations.
        dataset_folder : folder to store results and related data, such as initial settings.
        var_list : list of model's variable names to randomize for initial settings.
        prefix : string prefix for temporary files.
        how : string, one of {'concat', 'stack'}. Defines final shape of returned dataset.
            'concat' is used for training dataset;
            'stack' - for histogram dataset, so that we can easily build histograms for different initial settings
        settings_filename : file-name to save randomly generated initial settings.

        Returns
        -------
        dataset : np.array
         of shape (nb_settings * nb_trajectories, number-of-steps, number-of-species) If `how` == 'concat'
         or       (nb_settings, nb_trajectories, number-of-steps, number-of-species) If `how` == 'stack'

        """
    with open(model_name, 'r') as f:
        model_text = f.read()

    settings_fp = os.path.join(dataset_folder, settings_filename)
    settings = get_random_initial_settings(model_text, var_list, n_settings=nb_settings, sigm=1.0)

    with open(settings_fp, 'wb') as f:
        pickle.dump(settings, f)

    _perform_simulations(
        model_name,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_folder,
        prefix=prefix,
        settings_filename=settings_filename,
    )
    if how == 'concat':
        LOGGER.info(">>>> starting concatenate_simulations...")
        dataset = _concatenate_simulations(nb_settings, dataset_folder, prefix=prefix)
        LOGGER.info(">>>> done...")
    elif how == 'stack':
        LOGGER.info(">>>> starting stack_simulations...")
        dataset = _stack_simulations(nb_settings, dataset_folder, prefix=prefix)
        LOGGER.info(">>>> done...")
    else:
        raise ValueError("'how' accepts only two arguments: "
                         "'concat' and 'stack'.")
    return dataset


def _perform_simulations(
        model_fp,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_folder,
        prefix='partial_',
        settings_filename='settings.pickle',
):
    """Perform simulations, save intermediate results and initial settings to `dataset_folder`."""
    settings_fp = os.path.join(dataset_folder, settings_filename)
    with open(settings_fp, 'rb') as f:
        settings = pickle.load(f)

    count = (multiprocessing.cpu_count() // 3) * 2
    LOGGER.info(" ===== CPU Cores used for simulations: %s =====" % count)
    pool = multiprocessing.Pool(processes=count)

    with open(model_fp, 'r') as f:
        model_text = f.read()

    sim_params = kappy.SimulationParameter(timestep, f"[T] > {endtime}")

    kwargs = [(settings[n], n) for n in range(nb_settings)]

    task = partial(
        _single_simulation,
        model_text=model_text,
        sim_params=sim_params,
        nb_trajectories=nb_trajectories,
        dataset_folder=dataset_folder,
        prefix=prefix,
    )

    pool.starmap(task, kwargs)
    pool.close()
    return


def _single_simulation(
        settings,
        id_number,
        model_text,
        sim_params,
        nb_trajectories,
        dataset_folder,
        prefix
):
    """Helper single-thread function."""
    kappa_client = kappy.KappaStd()
    kappa_client.add_model_string(model_text)
    kappa_client.project_parse(**settings)
    data = []

    for i in range(nb_trajectories):
        kappa_client.simulation_start(sim_params)
        kappa_client.wait_for_simulation_stop()
        res = np.array(kappa_client.simulation_plot()['series'])[::-1]
        data.append(res)
        kappa_client.simulation_delete()
    kappa_client.shutdown()

    min_length = min([t.shape[0] for t in data])
    data = np.stack([t[:min_length] for t in data], axis=0)

    _save_simulation_data(data, dataset_folder, prefix, id_number)


def _find_var_value(model_text, param_name):
    """Find value of `param_name` variable in Kappa model definition."""
    lines = model_text.splitlines()
    patt = f'[\',\"]+{param_name}+[\',\"]'
    try:
        line = next(
            line for line in lines
            if line.startswith('%var:') and re.search(patt, line)
        )
        return float(line.split(' ')[2])
    except StopIteration:
        return None


def get_random_initial_settings(model_text, var_list, n_settings, sigm=0.7):
    """
    Produce random settings for variables in var_list. Returned settings can be
    directly fed to get_simulations. New are drawn based on corresponding initial
    values in the model_text:
        - low = max(0, initial_value * (1 - sigm))
        - high = initial_value * (1 + sigm)

    Parameters
    ----------
    model_text : text containing kappa model definition (content of model.ka).
    var_list : subset of model variables to produce random settings.
    n_settings : number of settings to produce
    sigm : float parameter to set upper and lower bounds to draw settings.

    Returns
    -------
    settings : list of dictionaries {var_name: value} of length n_settings.

    """
    initial_values = {name: _find_var_value(model_text, name) for name in var_list}
    print(initial_values)
    settings = []
    for _ in range(n_settings):
        s = dict()
        for name in var_list:
            low = max(0, initial_values[name] * (1 - sigm))
            high = initial_values[name] * (1 + sigm)
            s[name] = np.random.randint(low, high)
        settings.append(s)
    return settings
