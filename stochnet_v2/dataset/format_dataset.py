import argparse
import os
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--project_folder', type=str, required=True)
parser.add_argument('--timestep', type=float, required=True)
parser.add_argument('--dataset_id', type=int, required=True)
parser.add_argument('--nb_past_timesteps', type=int, default=1)
parser.add_argument('--positivity', type=str, default='true')
parser.add_argument('--test_fraction', type=float, default=0.125)
parser.add_argument('--save_format', type=str, default='hdf5', choices=['hdf5', 'tfrecord'])
parser.add_argument('--random_seed', type=int, default=23)
args = parser.parse_args()


def true_or_false(arg):
    arg_upper = str(arg).upper()
    if 'TRUE'.startswith(arg_upper):
        return True
    elif 'FALSE'.startswith(arg_upper):
        return False
    else:
        pass


def main():

    path = os.path.dirname(__file__)
    sys.path.append(os.path.join(path, '../..'))
    from stochnet_v2.utils.file_organisation import ProjectFileExplorer
    from stochnet_v2.dataset.dataset import DataTransformer

    np.random.seed(args.random_seed)

    project_explorer = ProjectFileExplorer(args.project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(args.timestep, args.dataset_id)

    dt = DataTransformer(dataset_explorer.dataset_fp)

    if args.save_format == 'hdf5':
        save_fn = dt.save_data_for_ml_hdf5
    elif args.save_format == 'tfrecord':
        save_fn = dt.save_data_for_ml_tfrecord
    else:
        raise ValueError(f"save_format `{args.save_format}` not recognized")

    positivity = true_or_false(args.positivity)

    save_fn(
        dataset_folder=dataset_explorer.dataset_folder,
        nb_past_timesteps=args.nb_past_timesteps,
        test_fraction=args.test_fraction,
        keep_timestamps=False,
        rescale=False,
        positivity=positivity,
        shuffle=True,
        slice_size=100,
        force_rewrite=True
    )

    save_fn(
        dataset_folder=dataset_explorer.dataset_folder,
        nb_past_timesteps=args.nb_past_timesteps,
        test_fraction=args.test_fraction,
        keep_timestamps=False,
        rescale=True,
        positivity=positivity,
        shuffle=True,
        slice_size=100,
        force_rewrite=True
    )


if __name__ == '__main__':
    main()

"""
python stochnet_v2/dataset/format_dataset.py \
       --project_folder='/home/dn/DATA/EGFR' \
       --timestep=0.5 \
       --dataset_id=4 \
       --nb_past_timesteps=1 \
       --positivity=true \
       --test_fraction=0.2 \
       --save_format='hdf5' \
       --random_seed=29
"""