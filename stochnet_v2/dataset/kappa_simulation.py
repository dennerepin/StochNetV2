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
    new_text = model_text
    for name in values_dict:
        new_text = _set_var_value(new_text, name, values_dict[name])
    return new_text


def _find_param_value(model_text, param_name):
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
    initial_values = {name: _find_param_value(model_text, name) for name in var_list}
    settings = []
    for _ in range(n_settings):
        s = dict()
        for name in var_list:
            low = max(0, initial_values[name] * (1 - sigm))
            high = initial_values[name] * (1 + sigm)
            s[name] = np.random.randint(low, high)
        settings.append(s)
    return settings
