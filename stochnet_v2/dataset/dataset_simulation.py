import argparse
import multiprocessing
import numpy as np
import os
import sys

from functools import partial
from gillespy import StochKitSolver
from importlib import import_module
from tqdm import tqdm
from time import time


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

    crn_module = import_module("stochnet_v2.CRN_models." + model_name)
    crn_class = getattr(crn_module, model_name)
    crn_instance = crn_class(endtime, timestep)

    count = (multiprocessing.cpu_count() // 3) * 2 + 1
    print(" ===== CPU Cores used for simulations: %s =====" % count)
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
    print(f"Saving to partial_dataset_filepath: {partial_dataset_filepath}")
    np.save(partial_dataset_filepath, data)
    print("Saved.")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, required=True)
    parser.add_argument('--timestep', type=float, required=True)
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--nb_settings', type=int, required=True)
    parser.add_argument('--nb_trajectories', type=int, required=True)
    parser.add_argument('--endtime', type=float, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=23)
    args = parser.parse_args()

    path = os.path.dirname(__file__)
    sys.path.append(os.path.join(path, '../..'))
    from stochnet_v2.utils.file_organisation import ProjectFileExplorer

    project_folder = args.project_folder
    timestep = args.timestep
    dataset_id = args.dataset_id
    nb_settings = args.nb_settings
    nb_trajectories = args.nb_trajectories
    endtime = args.endtime
    model_name = args.model_name
    random_seed = args.random_seed

    print(">>> START")
    start = time()

    np.random.seed(random_seed)

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)

    crn_module = import_module("stochnet_v2.CRN_models." + model_name)
    crn_class = getattr(crn_module, model_name)
    settings = crn_class.get_initial_settings(nb_settings)
    np.save(dataset_explorer.settings_fp, settings)

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
    np.save(dataset_explorer.dataset_fp, dataset)

    print(">>> DONE.")

    end = time()
    execution_time = end - start
    msg = f"Simulating {nb_trajectories} {model_name} " \
          f"trajectories for {nb_settings} different settings " \
          f"with endtime {endtime} took {execution_time} seconds.\n"\

    with open(dataset_explorer.log_fp, 'a') as f:
        f.write(msg)

    print(msg)


if __name__ == '__main__':
    main()


"""
python stochnet_v2/dataset/dataset_simulation.py \
       --project_folder '/home/dn/DATA/EGFR' \
       --timestep 0.2 \
       --dataset_id 1 \
       --nb_settings 2 \
       --nb_trajectories 5 \
       --endtime 10 \
       --model_name 'EGFR' \
       --random_seed 43
"""
