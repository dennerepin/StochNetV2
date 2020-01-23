import argparse
import os

from stochnet_v2.utils.evaluation import evaluate
from stochnet_v2.utils.file_organisation import ProjectFileExplorer


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, required=True)
    parser.add_argument('--timestep', type=float, required=True)
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--model_id', type=int, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--nb_past_timesteps', type=int, required=True)
    parser.add_argument('--distance_kind', type=str, default='dist', choices=['dist', 'iou'])
    parser.add_argument('--target_species_names', default='')
    parser.add_argument('--time_lag_range', default='10')
    parser.add_argument('--settings_idxs_to_save_histograms', default='0 1')

    args = parser.parse_args()

    project_folder = args.project_folder
    timestep = args.timestep
    dataset_id = args.dataset_id
    model_id = args.model_id
    model_name = args.model_name
    nb_past_timesteps = args.nb_past_timesteps
    distance_kind = args.distance_kind
    target_species_names = args.target_species_names.split(' ')
    target_species_names = target_species_names if target_species_names != [''] else []
    time_lag_range = args.time_lag_range
    time_lag_range = list(map(int, time_lag_range.split(' ')))
    settings_idxs_to_save_histograms = list(map(int, args.settings_idxs_to_save_histograms.split(' ')))

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)
    histogram_explorer = dataset_explorer.get_histogram_file_explorer(model_id, 0)
    nn_histogram_data_fp = os.path.join(histogram_explorer.model_histogram_folder, 'nn_histogram_data.npy')

    evaluate(
        model_name=model_name,
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        nb_past_timesteps=nb_past_timesteps,
        n_bins=100,
        distance_kind=distance_kind,
        with_timestamps=True,
        save_histograms=True,
        time_lag_range=time_lag_range,
        target_species_names=target_species_names,
        path_to_save_nn_traces=nn_histogram_data_fp,
        settings_idxs_to_save_histograms=settings_idxs_to_save_histograms,
    )


if __name__ == "__main__":
    main()


"""
python stochnet_v2/scripts/evaluate.py \
    --project_folder='/home/dn/DATA/EGFR' \
    --timestep=0.2 \
    --dataset_id=1 \
    --model_id=2 \
    --model_name='EGFR' \
    --nb_past_timesteps=1 \
    --distance_kind='iou' \
    --target_species_names='EGF R' \
    --time_lag_range='10 11' \
    --settings_idxs_to_save_histograms='0'
"""
