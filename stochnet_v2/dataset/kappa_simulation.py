import logging
import kappy
import multiprocessing
import numpy as np
import re
from functools import partial

logger = logging.getLogger('scripts.kappa_simulation')


def _single_simulation(sim_params, values_dict, model_text):
    if values_dict:
        model_text = set_initial_values(model_text, values_dict)
    kappa_client = kappy.KappaStd()
    kappa_client.add_model_string(model_text)
    kappa_client.project_parse()
    kappa_client.simulation_start(sim_params)
    kappa_client.wait_for_simulation_stop()
    res = kappa_client.simulation_plot()
    kappa_client.simulation_delete()
    kappa_client.shutdown()
    del kappa_client
    return np.array(res['series'])[::-1]


def get_simulations(model_text, n_simulations, sim_params, initial_settings=None):
    """
    Produces traces for kappa model.

    Parameters
    ----------
    model_text : text containing kappa model definition (content of model.ka).
    n_simulations : number of traces to simulate. If initial_settings is not None,
        then this amount of traces will be produced for each of settings.
    sim_params : kappy.SimulationParameter instance, defining time-resolution and simulation stop time.
    initial_settings : list of dictionaries of {var_name: value}, optional.

    Returns
    -------
    numpy array of traces of shape (n_settings * n_simulations, n_steps, n_observable_species + 1)

    """
    count = (multiprocessing.cpu_count() // 3) * 2
    pool = multiprocessing.Pool(processes=count)

    task = partial(_single_simulation, model_text=model_text)

    args = []

    if initial_settings:
        [args.extend([(sim_params, s) for _ in range(n_simulations)]) for s in initial_settings]
    else:
        args.extend([(sim_params, None) for _ in range(n_simulations)])

    traces = pool.starmap(task, args)
    pool.close()

    min_length = min([t.shape[0] for t in traces])
    return np.stack([t[:min_length] for t in traces], axis=0)


def _set_var_value(model_text, param_name, param_value):
    lines = model_text.splitlines()
    patt = f'[\',\"]+{param_name}+[\',\"]'
    try:
        idx = next(
            i
            for i, line in enumerate(lines)
            if line.startswith('%var:') and re.search(patt, line)
        )
    except StopIteration:
        logger.error(f"Parameter {param_name} not found. Ignored.")
        idx = None

    if idx:
        new_line = ' '.join(lines[idx].split(' ')[:2] + [str(param_value)])
        new_text = '\n'.join(lines[:idx] + [new_line] + lines[idx+1:])
        return new_text

    return model_text


def set_initial_values(model_text, values_dict):
    """
    Set values for variables of kappa model.

    Parameters
    ----------
    model_text : text containing kappa model definition (content of model.ka).
    values_dict : dictionary of {var_name: value}

    Returns
    -------
    new_text : edited model definition

    """
    new_text = model_text
    for name in values_dict:
        new_text = _set_var_value(new_text, name, values_dict[name])
    return new_text


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
