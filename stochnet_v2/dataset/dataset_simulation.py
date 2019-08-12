import multiprocessing
import numpy as np
import os
import sys

from functools import partial
from gillespy import StochKitSolver
from importlib import import_module
from tqdm import tqdm
from time import time

# path = os.path.dirname(__file__)
# sys.path.append(os.path.join(path, '../..'))
# from stochnet_v2.utils.file_organisation import ProjectFileExplorer


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
        print(">>>> starting concatenate_simulations...")
        dataset = concatenate_simulations(nb_settings, dataset_folder, prefix=prefix)
        print(">>>> done...")
    elif how == 'stack':
        print(">>>> starting stack_simulations...")
        dataset = stack_simulations(nb_settings, dataset_folder, prefix=prefix)
        print(">>>> done...")
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

    CRN_module = import_module("stochnet_v2.CRN_models." + model_name)
    CRN_class = getattr(CRN_module, model_name)
    CRN = CRN_class(endtime, timestep)

    count = multiprocessing.cpu_count() // 2
    print(" ===== CPU Cores used for simulations: %s =====" % count)
    pool = multiprocessing.Pool(processes=count)

    kwargs = [(settings[n], n) for n in range(nb_settings)]

    task = partial(
        single_simulation,
        CRN=CRN,
        nb_trajectories=nb_trajectories,
        dataset_folder=dataset_folder,
        prefix=prefix
    )

    pool.starmap(task, kwargs)
    return


def single_simulation(
        initial_values,
        id_number,
        CRN,
        nb_trajectories,
        dataset_folder,
        prefix,
):

    CRN.set_species_initial_value(initial_values)
    trajectories = CRN.run(
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
    print(f"Saving to partial_dataset_filepath: {partial_dataset_filepath}")
    with open(partial_dataset_filepath, 'wb') as file:
        np.save(file, data)
    print("Saved.")
    return


if __name__ == '__main__':
    path = os.path.dirname(__file__)
    sys.path.append(os.path.join(path, '../..'))
    from stochnet_v2.utils.file_organisation import ProjectFileExplorer

    print(">>> START")
    start = time()
    dataset_id = int(sys.argv[1])
    nb_settings = int(sys.argv[2])
    nb_trajectories = int(sys.argv[3])
    timestep = float(sys.argv[4])
    endtime = float(sys.argv[5])
    project_folder = str(sys.argv[6])
    model_name = str(sys.argv[7])

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep,
                                                                  dataset_id)

    CRN_module = import_module("stochnet_v2.CRN_models." + model_name)
    CRN_class = getattr(CRN_module, model_name)
    settings = CRN_class.get_initial_settings(nb_settings)
    settings_fp = os.path.join(dataset_explorer.dataset_folder, 'settings.npy')

    np.save(settings_fp, settings)
    nb_settings = settings.shape[0]

    print(f"Dataset folder: {dataset_explorer.dataset_folder}")

    dataset = build_simulation_dataset(
        model_name,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_explorer.dataset_folder,
        how='concat'
    )

    with open(dataset_explorer.dataset_fp, 'wb') as f:
        np.save(f, dataset)

    print(">>> DONE.")

    end = time()
    execution_time = end - start
    msg = f"Simulating {nb_trajectories} {model_name} " \
          f"trajectories for {nb_settings} different settings " \
          f"with endtime {endtime} took {execution_time} seconds.\n"\

    with open(dataset_explorer.log_fp, 'a') as f:
        f.write(msg)

    print(msg)


# python stochnet_v2/dataset/dataset_simulation.py 1000 10 50 0.2 20 '/home/dn/Documents/tmp/tmp_simulations' 'EGFR'
