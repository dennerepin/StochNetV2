import kappy
import logging
import multiprocessing
import numpy as np
import os
import pickle
import re
from functools import partial

from stochnet_v2.dataset.simulation_gillespy import concatenate_simulations
from stochnet_v2.dataset.simulation_gillespy import stack_simulations
from stochnet_v2.dataset.simulation_gillespy import save_simulation_data

LOGGER = logging.getLogger('scripts.kappa_simulation')


def build_simulation_dataset(
        model_name,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_folder,
        prefix='partial_',
        how='concat',
        **kwargs,
):
    perform_simulations(
        model_name,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_folder,
        prefix=prefix,
        **kwargs,
    )
    if how == 'concat':
        LOGGER.info(">>>> starting concatenate_simulations...")
        dataset = concatenate_simulations(nb_settings, dataset_folder, prefix=prefix)
        LOGGER.info(">>>> done...")
    elif how == 'stack':
        LOGGER.info(">>>> starting stack_simulations...")
        dataset = stack_simulations(nb_settings, dataset_folder, prefix=prefix)
        LOGGER.info(">>>> done...")
    else:
        raise ValueError("'how' accepts only two arguments: "
                         "'concat' and 'stack'.")
    return dataset


def perform_simulations(
        model_fp,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_folder,
        prefix='partial_',
        settings_filename='settings.pickle',
):
    settings_fp = os.path.join(dataset_folder, settings_filename)
    with open(settings_fp, 'rb') as f:
        settings = pickle.load(f)

    count = (multiprocessing.cpu_count() // 3) * 2 + 1
    LOGGER.info(" ===== CPU Cores used for simulations: %s =====" % count)
    pool = multiprocessing.Pool(processes=count)

    with open(model_fp, 'r') as f:
        model_text = f.read()

    sim_params = kappy.SimulationParameter(timestep, f"[T] > {endtime}")

    kwargs = [(settings[n], n) for n in range(nb_settings)]

    task = partial(
        single_simulation,
        model_text=model_text,
        sim_params=sim_params,
        nb_trajectories=nb_trajectories,
        dataset_folder=dataset_folder,
        prefix=prefix,
    )

    pool.starmap(task, kwargs)
    pool.close()
    return


def single_simulation(
        settings,
        id_number,
        model_text,
        sim_params,
        nb_trajectories,
        dataset_folder,
        prefix
):
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

    print(f' - {id_number} Data shape: {data.shape}')
    save_simulation_data(data, dataset_folder, prefix, id_number)


def _find_var_value(model_text, param_name):
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
    Produces random settings for variables in var_list. Returned settings can be
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
    settings = []
    for _ in range(n_settings):
        s = dict()
        for name in var_list:
            low = max(0, initial_values[name] * (1 - sigm))
            high = initial_values[name] * (1 + sigm)
            s[name] = np.random.randint(low, high)
        settings.append(s)
    return settings
