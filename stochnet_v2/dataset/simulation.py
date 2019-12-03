import logging
import multiprocessing
import numpy as np
import os

from functools import partial
from gillespy import StochKitSolver
from importlib import import_module
from tqdm import tqdm

LOGGER = logging.getLogger('dataset.simulation')


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


def concatenate_simulations(nb_settings, dataset_folder, prefix='partial_'):
    for i in tqdm(range(nb_settings)):
        partial_dataset_filename = str(prefix) + str(i) + '.npy'
        partial_dataset_filepath = os.path.join(
            dataset_folder,
            partial_dataset_filename
        )
        with open(partial_dataset_filepath, 'rb') as f:
            partial_dataset = np.load(f)
        if i == 0:
            final_dataset = partial_dataset
        else:
            final_dataset = np.concatenate(
                (final_dataset, partial_dataset),
                axis=0,
            )
        os.remove(partial_dataset_filepath)
    return final_dataset


def stack_simulations(nb_settings, dataset_folder, prefix='partial_'):
    for i in tqdm(range(nb_settings)):
        partial_dataset_filename = str(prefix) + str(i) + '.npy'
        partial_dataset_filepath = os.path.join(
            dataset_folder,
            partial_dataset_filename
        )
        with open(partial_dataset_filepath, 'rb') as f:
            partial_dataset = np.load(f)
        if i == 0:
            final_dataset = partial_dataset[np.newaxis, ...]
        else:
            final_dataset = np.concatenate(
                (final_dataset, partial_dataset[np.newaxis, ...]),
                axis=0,
            )
        os.remove(partial_dataset_filepath)
    return final_dataset


def perform_simulations(
        model_name,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_folder,
        prefix='partial_',
        settings_filename='settings.npy',
):
    settings_fp = os.path.join(dataset_folder, settings_filename)
    settings = np.load(settings_fp)

    crn_module = import_module("stochnet_v2.CRN_models." + model_name)
    crn_class = getattr(crn_module, model_name)
    crn_instance = crn_class(endtime, timestep)

    count = (multiprocessing.cpu_count() // 3) * 2 + 1
    LOGGER.info(" ===== CPU Cores used for simulations: %s =====" % count)
    pool = multiprocessing.Pool(processes=count)

    kwargs = [(settings[n], n) for n in range(nb_settings)]

    task = partial(
        single_simulation,
        crn_instance=crn_instance,
        nb_trajectories=nb_trajectories,
        dataset_folder=dataset_folder,
        prefix=prefix
    )

    pool.starmap(task, kwargs)
    pool.close()
    return


def single_simulation(
        initial_values,
        id_number,
        crn_instance,
        nb_trajectories,
        dataset_folder,
        prefix,
):

    crn_instance.set_species_initial_value(initial_values)
    trajectories = crn_instance.run(
        number_of_trajectories=nb_trajectories,
        solver=StochKitSolver,
        show_labels=False
    )
    data = np.array(trajectories)
    save_simulation_data(data, dataset_folder, prefix, id_number)


def save_simulation_data(
        data,
        dataset_folder,
        prefix,
        id_number,
):
    partial_dataset_filename = str(prefix) + str(id_number) + '.npy'
    partial_dataset_filepath = os.path.join(dataset_folder,
                                            partial_dataset_filename)
    LOGGER.info(f"Saving to partial_dataset_filepath: {partial_dataset_filepath}")
    np.save(partial_dataset_filepath, data)
    LOGGER.info("Saved.")
    return
