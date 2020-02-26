import argparse
import h5py
import numpy as np
import os
from time import time
from importlib import import_module
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.dataset.simulation_gillespy import build_simulation_dataset


def get_histogram_settings(
        nb_histogram_settings,
        train_test_data_fp,
):
    """Randomly selects a subset of the x dataset to be used as initial setup
    in the construction of the histogram dataset.
    """
    data_file = h5py.File(train_test_data_fp, 'r')
    x_data = data_file['x']
    nb_samples = x_data.shape[0]
    settings_idxs = sorted(np.random.randint(
        low=0,
        high=nb_samples - 1,
        size=nb_histogram_settings,
    ))
    settings = x_data[settings_idxs, 0, :]
    data_file.close()
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, required=True)
    parser.add_argument('--timestep', type=float, required=True)
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--nb_settings', type=int, required=True)
    parser.add_argument('--nb_trajectories', type=int, required=True)
    parser.add_argument('--endtime', type=float, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--params_to_randomize', required=True, default='')
    parser.add_argument('--random_seed', type=int, default=23)
    args = parser.parse_args()

    project_folder = args.project_folder
    timestep = args.timestep
    dataset_id = args.dataset_id
    nb_settings = args.nb_settings
    nb_trajectories = args.nb_trajectories
    endtime = args.endtime
    model_name = args.model_name
    params_to_randomize = args.params_to_randomize.split(' ')
    params_to_randomize = params_to_randomize if params_to_randomize != [''] else []
    random_seed = args.random_seed

    start = time()

    np.random.seed(random_seed)

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(
        timestep,
        dataset_id
    )

    # settings = get_histogram_settings(
    #     nb_settings,
    #     # dataset_explorer.train_fp,
    #     dataset_explorer.test_fp,
    # )

    crn_module = import_module("stochnet_v2.CRN_models." + model_name)
    crn_class = getattr(crn_module, model_name)
    settings = crn_class.get_initial_settings(nb_settings)

    np.save(dataset_explorer.histogram_settings_fp, settings)

    histogram_dataset = build_simulation_dataset(
        model_name,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_explorer.dataset_folder,
        params_to_randomize=params_to_randomize,
        prefix='histogram_partial_',
        how='stack',
        settings_filename=os.path.basename(dataset_explorer.histogram_settings_fp),
    )
    np.save(dataset_explorer.histogram_dataset_fp, histogram_dataset)

    end = time()
    execution_time = end - start

    with open(dataset_explorer.log_fp, 'a') as file:
        file.write(
            f"\n\nSimulating {nb_trajectories} {model_name} histogram trajectories "
            f"for {nb_settings} different settings until {endtime} "
            f"took {execution_time} seconds.\n"
        )


if __name__ == '__main__':
    main()


"""
python stochnet_v2/scripts/simulate_histogram_data_gillespy.py \
       --project_folder '/home/dn/DATA/SIR' \
       --timestep 0.5 \
       --dataset_id 3 \
       --nb_settings 2 \
       --nb_trajectories 25 \
       --endtime 10 \
       --model_name 'SIR' \
       --params_to_randomize 'beta gamma' \
       --random_seed 43
"""