import argparse
import h5py
import numpy as np
import os
import sys

from time import time

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
    path = os.path.dirname(__file__)
    sys.path.append(os.path.join(path, '../..'))
    from stochnet_v2.utils.file_organisation import ProjectFileExplorer
    from stochnet_v2.dataset.dataset_simulation import build_simulation_dataset

    start = time()

    np.random.seed(args.random_seed)

    project_explorer = ProjectFileExplorer(args.project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(
        args.timestep,
        args.dataset_id
    )

    settings = get_histogram_settings(
        args.nb_settings,
        dataset_explorer.train_fp
    )
    np.save(dataset_explorer.histogram_settings_fp, settings)

    histogram_dataset = build_simulation_dataset(
        args.model_name,
        args.nb_settings,
        args.nb_trajectories,
        args.timestep,
        args.endtime,
        dataset_explorer.dataset_folder,
        prefix='histogram_partial_',
        how='stack',
        settings_filename=os.path.basename(dataset_explorer.histogram_settings_fp),
    )
    np.save(dataset_explorer.histogram_dataset_fp, histogram_dataset)

    end = time()
    execution_time = end - start

    with open(dataset_explorer.log_fp, 'a') as file:
        file.write(
            f"Simulating {args.nb_trajectories} {args.model_name} histogram trajectories "
            f"for {args.nb_settings} different settings until {args.endtime} "
            f"took {execution_time} seconds.\n"
        )


if __name__ == '__main__':
    main()


"""
python stochnet_v2/dataset/histogram_dataset_simulation.py \
       --project_folder='/home/dn/DATA/Gene' \
       --timestep=400 \
       --dataset_id=3 \
       --nb_settings=2 \
       --nb_trajectories=5 \
       --endtime=10000 \
       --model_name='Gene' \
       --random_seed=43
"""
