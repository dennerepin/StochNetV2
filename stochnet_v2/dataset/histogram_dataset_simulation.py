import h5py
import numpy as np
import os
import sys

from time import time

path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, '../..'))
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.dataset.dataset_simulation import build_simulation_dataset


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


if __name__ == '__main__':
    start = time()

    source_dataset_id = int(sys.argv[1])
    source_dataset_timestep = float(sys.argv[2])
    histogram_dataset_id = int(sys.argv[3])
    nb_histogram_settings = int(sys.argv[4])
    nb_trajectories = int(sys.argv[5])
    histogram_timestep = float(sys.argv[6])
    endtime = float(sys.argv[7])
    project_folder = str(sys.argv[8])
    model_name = str(sys.argv[9])
    random_seed = int(sys.argv[10])

    np.random.seed(random_seed)

    project_explorer = ProjectFileExplorer(project_folder)
    source_dataset_explorer = project_explorer.get_dataset_file_explorer(
        source_dataset_timestep,
        source_dataset_id
    )

    settings = get_histogram_settings(
        nb_histogram_settings,
        source_dataset_explorer.train_fp
    )

    histogram_dataset_explorer = project_explorer.get_dataset_file_explorer(
        histogram_timestep,
        histogram_dataset_id
    )
    with open(histogram_dataset_explorer.histogram_settings_fp, 'wb') as file:
        np.save(file, settings)

    histogram_dataset = build_simulation_dataset(
        model_name,
        nb_histogram_settings,
        nb_trajectories,
        histogram_timestep,
        endtime,
        histogram_dataset_explorer.dataset_folder,
        prefix='histogram_partial_',
        how='stack',
        settings_filename=os.path.basename(histogram_dataset_explorer.histogram_settings_fp),
    )

    with open(histogram_dataset_explorer.histogram_dataset_fp, 'wb') as file:
        np.save(file, histogram_dataset)

    end = time()
    execution_time = end - start

    with open(histogram_dataset_explorer.log_fp, 'a') as file:
        file.write(
            f"Simulating {nb_trajectories} {model_name} histogram trajectories "
            f"for {nb_histogram_settings} different settings until {endtime} "
            f"took {execution_time} seconds.\n"
        )


# python stochnet_v2/dataset/histogram_dataset_simulation.py 1000 0.2 1000 10 200 0.2 20 '/home/dn/Documents/tmp/tmp_simulations' 'EGFR' 1