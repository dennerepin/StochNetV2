import argparse
import numpy as np
import os
import pickle
from time import time

from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.dataset.simulation_kappy import build_simulation_dataset
from stochnet_v2.dataset.simulation_kappy import get_random_initial_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, required=True)
    parser.add_argument('--timestep', type=float, required=True)
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--var_list', type=str, required=True,
                        help='string of space-separated variable names to randomize')
    parser.add_argument('--nb_settings', type=int, required=True)
    parser.add_argument('--nb_trajectories', type=int, required=True)
    parser.add_argument('--endtime', type=float, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=23)
    args = parser.parse_args()

    project_folder = args.project_folder
    timestep = args.timestep
    dataset_id = args.dataset_id
    var_list = args.var_list.split(' ')
    nb_settings = args.nb_settings
    nb_trajectories = args.nb_trajectories
    endtime = args.endtime
    model_name = args.model_name
    random_seed = args.random_seed

    start = time()

    np.random.seed(random_seed)

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(
        timestep,
        dataset_id
    )

    settings_filename = 'histogram_settings.pickle'

    settings_fp = os.path.join(dataset_explorer.dataset_folder, settings_filename)
    model_fp = os.path.join(project_folder, f'{model_name}.ka')

    with open(model_fp, 'r') as f:
        model_text = f.read()

    settings = get_random_initial_settings(model_text, var_list, n_settings=nb_settings, sigm=1.0)

    with open(settings_fp, 'wb') as f:
        pickle.dump(settings, f)

    histogram_dataset = build_simulation_dataset(
        model_fp,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_explorer.dataset_folder,
        prefix='histogram_partial_',
        how='stack',
        settings_filename=settings_filename,
    )
    np.save(dataset_explorer.histogram_dataset_fp, histogram_dataset)

    end = time()
    execution_time = end - start

    with open(dataset_explorer.log_fp, 'a') as file:
        file.write(
            f"Simulating {nb_trajectories} {model_name} histogram trajectories "
            f"for {nb_settings} different settings until {endtime} "
            f"took {execution_time} seconds.\n"
        )


if __name__ == '__main__':
    main()


"""
python stochnet_v2/scripts/simulate_histogram_data_kappy.py \
       --project_folder='/home/dn/DATA/LST_kappa_loop' \
       --timestep=100.0 \
       --dataset_id=2 \
       --var_list='a_add p0' \
       --nb_settings=2 \
       --nb_trajectories=10 \
       --endtime=10000 \
       --model_name='LST_kappa_loop' \
       --random_seed=44
"""