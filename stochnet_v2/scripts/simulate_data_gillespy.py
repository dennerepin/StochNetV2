import argparse
import logging
import numpy as np
from importlib import import_module
from time import time

from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.dataset.simulation_gillespy import build_simulation_dataset

LOGGER = logging.getLogger('scripts.simulate_data')


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
    random_seed = args.random_seed
    params_to_randomize = args.params_to_randomize.split(' ')
    params_to_randomize = params_to_randomize if params_to_randomize != [''] else []

    LOGGER.info(">>> START")
    start = time()

    np.random.seed(random_seed)

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)

    crn_module = import_module("stochnet_v2.CRN_models." + model_name)
    crn_class = getattr(crn_module, model_name)
    settings = crn_class.get_initial_settings(nb_settings)
    np.save(dataset_explorer.settings_fp, settings)

    LOGGER.info(f"Dataset folder: {dataset_explorer.dataset_folder}")

    dataset = build_simulation_dataset(
        model_name,
        nb_settings,
        nb_trajectories,
        timestep,
        endtime,
        dataset_explorer.dataset_folder,
        params_to_randomize=params_to_randomize,
        how='concat'
    )
    np.save(dataset_explorer.dataset_fp, dataset)

    LOGGER.info(">>> DONE.")

    end = time()
    execution_time = end - start
    msg = f"\n\nSimulating {nb_trajectories} {model_name} " \
          f"trajectories for {nb_settings} different settings " \
          f"with endtime {endtime} took {execution_time} seconds.\n"\

    with open(dataset_explorer.log_fp, 'a') as f:
        f.write(msg)

    LOGGER.info(msg)


if __name__ == '__main__':
    main()


"""
python stochnet_v2/scripts/simulate_data_gillespy.py \
       --project_folder '/home/dn/DATA/SIR' \
       --timestep 0.5 \
       --dataset_id 3 \
       --nb_settings 2 \
       --nb_trajectories 5 \
       --endtime 10 \
       --model_name 'SIR' \
       --params_to_randomize 'beta gamma' \
       --random_seed 43
"""
