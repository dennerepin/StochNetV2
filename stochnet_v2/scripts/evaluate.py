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
    parser.add_argument('--nb_randomized_params', type=int, required=True)
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
    nb_randomized_params = args.nb_randomized_params
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
        nb_randomized_params=nb_randomized_params,
        nb_past_timesteps=nb_past_timesteps,
        n_bins=200,
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
    --project_folder='/home/dn/DATA/SIR' \
    --timestep=0.5 \
    --dataset_id=3 \
    --model_id=3002 \
    --model_name='SIR' \
    --nb_past_timesteps=1 \
    --nb_randomized_params=2 \
    --distance_kind='dist' \
    --target_species_names='S I' \
    --time_lag_range='1 3 5' \
    --settings_idxs_to_save_histograms='0'
"""
