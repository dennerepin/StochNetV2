import argparse
import numpy as np
from time import time

from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.dataset.dataset import DataTransformer
from stochnet_v2.utils.util import str_to_bool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, required=True)
    parser.add_argument('--timestep', type=float, required=True)
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--nb_past_timesteps', type=int, default=1)
    parser.add_argument('--nb_randomized_params', type=int, required=True)
    parser.add_argument('--positivity', type=str, default='true')
    parser.add_argument('--test_fraction', type=float, default=0.125)
    parser.add_argument('--save_format', type=str, default='hdf5', choices=['hdf5', 'tfrecord'])
    parser.add_argument('--random_seed', type=int, default=23)
    args = parser.parse_args()

    project_folder = args.project_folder
    timestep = args.timestep
    dataset_id = args.dataset_id
    nb_past_timesteps = args.nb_past_timesteps
    nb_randomized_params = args.nb_randomized_params
    positivity = args.positivity
    test_fraction = args.test_fraction
    save_format = args.save_format
    random_seed = args.random_seed

    np.random.seed(random_seed)

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)

    dt = DataTransformer(
        dataset_explorer.dataset_fp,
        with_timestamps=True,
        nb_randomized_params=nb_randomized_params
    )

    if save_format == 'hdf5':
        save_fn = dt.save_data_for_ml_hdf5
    # elif save_format == 'tfrecord':
    #     save_fn = dt.save_data_for_ml_tfrecord
    else:
        raise ValueError(f"save_format `{save_format}` not recognized. Use 'hdf5'.")

    positivity = str_to_bool(positivity)

    start = time()

    save_fn(
        dataset_folder=dataset_explorer.dataset_folder,
        nb_past_timesteps=nb_past_timesteps,
        test_fraction=test_fraction,
        keep_timestamps=False,
        rescale=False,
        positivity=positivity,
        shuffle=True,
        slice_size=100,
        force_rewrite=True
    )

    save_fn(
        dataset_folder=dataset_explorer.dataset_folder,
        nb_past_timesteps=nb_past_timesteps,
        test_fraction=test_fraction,
        keep_timestamps=False,
        rescale=True,
        positivity=positivity,
        shuffle=True,
        slice_size=100,
        force_rewrite=True
    )

    end = time()
    execution_time = end - start
    msg = f"\n\nFormatting dataset into {save_format} files took {execution_time} seconds.\n" \
          f"\tnb_past_timesteps={nb_past_timesteps},\n" \
          f"\ttest_fraction={test_fraction},\n" \
          f"\tpositivity={positivity},\n" \

    with open(dataset_explorer.log_fp, 'a') as f:
        f.write(msg)


if __name__ == '__main__':
    main()


"""
python stochnet_v2/scripts/format_data_for_training.py \
       --project_folder='/home/dn/DATA/SIR' \
       --timestep=0.5 \
       --dataset_id=3 \
       --nb_past_timesteps=1 \
       --nb_randomized_params=2 \
       --positivity=true \
       --test_fraction=0.2 \
       --save_format='hdf5' \
       --random_seed=29
"""