import logging
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import os
from functools import partial
from importlib import import_module
from time import time
from tqdm import tqdm

from stochnet_v2.static_classes.model import StochNet
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.utils.util import maybe_create_dir


LOGGER = logging.getLogger('utils.evaluation')


def get_gillespy_histogram_data(
        project_folder,
        timestep,
        dataset_id,
):
    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)
    histogram_data = np.load(dataset_explorer.histogram_dataset_fp)
    return histogram_data


def get_nn_histogram_data(
        project_folder,
        timestep,
        dataset_id,
        model_id,
        nb_past_timesteps,
        nb_features,
        nb_randomized_params,
        n_steps,
        n_traces_per_setting,
        path_to_save_generated_data=None,
        add_timestamps=True,
        keep_params=False,
):
    nn = StochNet(
        nb_past_timesteps=nb_past_timesteps,
        nb_features=nb_features,
        nb_randomized_params=nb_randomized_params,
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        mode='inference'
    )

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)
    histogram_data = np.load(dataset_explorer.histogram_dataset_fp)
    initial_settings = histogram_data[:, 0, 0:nb_past_timesteps, -(nb_features+nb_randomized_params):]

    LOGGER.info("Start generating NN traces")
    cnt = 0
    while True:
        if cnt > 10:
            LOGGER.error(f"Failed to generate NN traces after {cnt} attempts...")
            break
        try:
            traces = nn.generate_traces(
                initial_settings,
                n_steps=n_steps,
                n_traces=n_traces_per_setting,
                curr_state_rescaled=False,
                round_result=True,
                add_timestamps=add_timestamps,
                keep_params=keep_params,
            )
            LOGGER.info(f"Done. generated data shape: {traces.shape}")
            if path_to_save_generated_data:
                np.save(path_to_save_generated_data, traces)
            return traces
        except:
            LOGGER.warning("Oops... trying again")
            cnt += 1


def _get_data_bounds(data, with_timestamps=True):
    # data ~ [n_settings, n_traces, n_steps, n_features]
    if with_timestamps:
        data = np.delete(data, 0, axis=-1)

    lower = np.min(data, axis=(0, 1, 2))
    upper = np.max(data, axis=(0, 1, 2))
    return list(zip(lower, upper))


def _get_histograms(
        data,
        time_lag,
        n_bins,
        target_species_idxs=None,
        histogram_bounds=None,
        with_timestamps=True
):
    # data ~ [n_settings, n_traces, n_steps, n_features]
    # ranges ~ [n_features]
    if with_timestamps:
        data = np.delete(data, 0, axis=-1)
    if target_species_idxs:
        data = data[..., target_species_idxs]

    n_settings, n_traces, n_steps, n_features = data.shape

    if histogram_bounds is not None:
        if len(histogram_bounds) != n_features:
            raise ValueError(
                f"Number of histogram ranges is not equal to the number of features: "
                f"{len(histogram_bounds)} and {n_features}"
            )
    all_settings_histograms = []

    for setting_idx in range(n_settings):
        setting_histograms = []

        for feature_idx in range(n_features):
            bins, edges = np.histogram(
                data[setting_idx, :, time_lag, feature_idx],
                bins=n_bins,
                range=histogram_bounds[feature_idx] if histogram_bounds else None,
            )
            bins = bins / np.sum(bins)
            setting_histograms.append((edges[1:], bins))

        all_settings_histograms.append(setting_histograms)

    return np.array(all_settings_histograms)


def _histogram_distance(histograms_1, histograms_2):
    if not histograms_1.shape == histograms_2.shape:
        raise ValueError("histogram shapes are not equal")
    n_settings, n_species, _, n_bins = histograms_2.shape
    distance = np.abs(histograms_1[:, :, 1, :] - histograms_2[:, :, 1, :])
    distance = np.sum(distance, axis=(0, -1)) / n_settings

    return distance


def _iou_distance(histograms_1, histograms_2):
    if not histograms_1.shape == histograms_2.shape:
        raise ValueError("histogram shapes are not equal")
    n_settings, n_species, _, n_bins = histograms_2.shape

    intersection = np.minimum(histograms_1[:, :, 1, :], histograms_2[:, :, 1, :])
    intersection = np.sum(intersection, axis=(-1))

    union = np.maximum(histograms_1[:, :, 1, :], histograms_2[:, :, 1, :])
    union = np.sum(union, axis=(-1))

    iou_dist = 1 - np.mean(intersection / union, axis=0)

    return iou_dist


def get_distance(
        time_lag,
        data_1,
        data_2,
        n_bins=100,
        target_species_idxs=None,
        histogram_bounds=None,
        with_timestamps=True,
        kind='dist',
        return_histograms=True,
):
    if histogram_bounds is None:
        bounds_1 = np.array(_get_data_bounds(data_1, with_timestamps))
        bounds_2 = np.array(_get_data_bounds(data_2, with_timestamps))
        lower = np.minimum(bounds_1[:, 0], bounds_2[:, 0])
        upper = np.maximum(bounds_1[:, 1], bounds_2[:, 1])
        histogram_bounds = list(zip(lower, upper))

        if target_species_idxs:
            histogram_bounds = [histogram_bounds[idx] for idx in target_species_idxs]

    histograms_1 = _get_histograms(
        data_1,
        time_lag,
        n_bins,
        target_species_idxs=target_species_idxs,
        histogram_bounds=histogram_bounds,
        with_timestamps=with_timestamps,
    )
    histograms_2 = _get_histograms(
        data_2,
        time_lag,
        n_bins,
        target_species_idxs=target_species_idxs,
        histogram_bounds=histogram_bounds,
        with_timestamps=with_timestamps,
    )
    if kind == 'dist':
        distance_fn = _histogram_distance
    elif kind == 'iou':
        distance_fn = _iou_distance
    else:
        raise ValueError("`kind` parameter unrecognized: {kind}. Should be one of: 'dist', 'iou'")

    distance = distance_fn(histograms_1, histograms_2)

    res = (distance, histograms_1, histograms_2) if return_histograms else distance
    return res


def evaluate(
        model_name,
        project_folder,
        timestep,
        dataset_id,
        model_id,
        nb_randomized_params,
        nb_past_timesteps=1,
        n_bins=100,
        distance_kind='iou',
        with_timestamps=True,
        save_histograms=True,
        time_lag_range=None,
        target_species_names=None,
        path_to_save_nn_traces=None,
        settings_idxs_to_save_histograms=None
):
    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)

    histogram_data = np.load(dataset_explorer.histogram_dataset_fp)
    histogram_data = histogram_data[..., :-nb_randomized_params]

    n_settings, n_traces, n_steps, n_species = histogram_data.shape
    if with_timestamps:
        n_species = n_species - 1

    CRN_module = import_module("stochnet_v2.CRN_models." + model_name)
    CRN_class = getattr(CRN_module, model_name)
    all_species_names = CRN_class.get_species_names()

    if len(all_species_names) != n_species:
        raise ValueError(
            f"Histogram data has {histogram_data.shape[-1]} species "
            f"({'with' if with_timestamps else 'without'} timesteps), "
            f"but CRN class {CRN_class.__name__} has {len(all_species_names)}."
        )

    target_species_names = target_species_names or all_species_names
    target_species_idxs = [all_species_names.index(name) for name in target_species_names]

    start = time()
    traces = get_nn_histogram_data(
        project_folder,
        timestep,
        dataset_id,
        model_id=model_id,
        nb_past_timesteps=nb_past_timesteps,
        nb_features=n_species,
        nb_randomized_params=nb_randomized_params,
        n_steps=n_steps-1,
        n_traces_per_setting=n_traces,
        path_to_save_generated_data=path_to_save_nn_traces,
        add_timestamps=with_timestamps,
        keep_params=False,
    )
    end = time()
    LOGGER.info(f"Took {end - start:.1f} seconds")
    with open(dataset_explorer.log_fp, 'a') as file:
        file.write(
            f"Simulating NN {n_traces} {model_name}, model_id={model_id} histogram trajectories "
            f"for {n_settings} different settings until {int(timestep * n_steps)}({n_steps} steps) "
            f"took {end - start:.1f} seconds.\n"
        )

    count = (multiprocessing.cpu_count() // 4) * 3 + 1
    pool = multiprocessing.Pool(processes=count)

    task = partial(
        get_distance,
        data_1=histogram_data,
        data_2=traces,
        n_bins=n_bins,
        with_timestamps=with_timestamps,
        target_species_idxs=target_species_idxs,
        histogram_bounds=None,
        kind=distance_kind,
        return_histograms=False,
    )

    LOGGER.info(f"Start calculating distances for different time-lags, using {count} CPU cores for multiprocessing")
    start = time()
    time_lags = list(range(n_steps - 1))

    species_distances = pool.map(task, time_lags)
    end = time()
    LOGGER.info(f"Took {end - start:.1f} seconds")

    mean_distances = [np.mean(dist_i) for dist_i in species_distances]
    species_distances = np.array(species_distances)

    histogram_explorer = dataset_explorer.get_histogram_file_explorer(model_id=model_id, nb_steps=0)
    mean_dist_fig_path = os.path.join(histogram_explorer.histogram_folder, os.path.pardir, f'mean_{distance_kind}')
    spec_dist_fig_path = os.path.join(histogram_explorer.histogram_folder, os.path.pardir, f'spec_{distance_kind}')

    fig = plt.figure(figsize=(12, 8))
    plt.title(f"Mean {distance_kind} distance (averaged over all species and {n_settings} settings)")
    plt.plot(mean_distances)
    plt.xlabel('time lag')
    plt.ylabel(f'distance')
    plt.savefig(mean_dist_fig_path)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    plt.title(f"Mean {distance_kind} distances (averaged over {n_settings} settings)")
    for i in range(species_distances.shape[-1]):
        plt.plot(species_distances[:, i], label=target_species_names[i])
    plt.xlabel('time lag')
    plt.ylabel(f'distance')
    plt.legend()
    plt.savefig(spec_dist_fig_path)
    plt.close(fig)

    if save_histograms:

        if settings_idxs_to_save_histograms is None:
            settings_idxs_to_save_histograms = [0]

        distance_fn = _histogram_distance if distance_kind == 'l1' else _iou_distance
        time_lag_range = time_lag_range or list(range(5, n_steps - 1, 10))

        LOGGER.info(
            f"Start building histograms for different settings: {settings_idxs_to_save_histograms}\n"
            f"and time-lags: {time_lag_range}"
        )
        start = time()

        for time_lag in tqdm(time_lag_range):

            histogram_explorer = dataset_explorer.get_histogram_file_explorer(model_id=model_id, nb_steps=time_lag)

            self_dist = get_distance(
                data_1=histogram_data[:, :n_traces // 2, ...],
                data_2=histogram_data[:, -n_traces // 2:, ...],
                time_lag=time_lag,
                n_bins=n_bins,
                with_timestamps=with_timestamps,
                target_species_idxs=target_species_idxs,
                histogram_bounds=None,
                kind=distance_kind,
                return_histograms=False,
            )

            species_distances, histograms_1, histograms_2 = get_distance(
                data_1=histogram_data,
                data_2=traces,
                time_lag=time_lag,
                n_bins=n_bins,
                with_timestamps=with_timestamps,
                target_species_idxs=target_species_idxs,
                histogram_bounds=None,
                kind=distance_kind,
                return_histograms=True,
            )
            self_dist_dict = {
                name: self_dist[idx]
                for idx, name in enumerate(target_species_names)
            }
            dist_dict = {
                name: species_distances[idx]
                for idx, name in enumerate(target_species_names)
            }
            with open(histogram_explorer.log_fp, 'w') as f:
                f.write(
                    f"Dataset mean self-distance ({distance_kind}): {np.mean(self_dist):.4f}\n"
                    f"Mean histogram distance ({distance_kind}): {np.mean(species_distances):.4f}\n"
                )

                f.write(f"\nDataset self-distances ({distance_kind}):\n")
                for k, v in self_dist_dict.items():
                    f.write(f"\t{k}: {v:.4f}\n")

                f.write(f"\nSpecies histogram distances ({distance_kind}):\n")
                for k, v in dist_dict.items():
                    f.write(f"\t{k}: {v:.4f}\n")

            for setting_idx in settings_idxs_to_save_histograms:

                for species_idx in range(len(target_species_idxs)):

                    curr_setting_distance = distance_fn(
                        histograms_1[setting_idx:setting_idx + 1, species_idx:species_idx + 1],
                        histograms_2[setting_idx:setting_idx + 1, species_idx:species_idx + 1]
                    )

                    save_path = os.path.join(
                        histogram_explorer.histogram_folder,
                        f'setting_{setting_idx}',
                        f'{target_species_names[species_idx]}'
                    )
                    maybe_create_dir(os.path.dirname(save_path))

                    fig = plt.figure(figsize=(12, 7))
                    plt.title(
                        f"{target_species_names[species_idx]}: "
                        f"{distance_kind}: {curr_setting_distance}, "
                        f"mean ({n_settings} settings): {species_distances[species_idx]:.4f}"
                    )
                    plt.plot(*histograms_1[setting_idx, species_idx], '-', label='gillespy')
                    plt.plot(*histograms_2[setting_idx, species_idx], '-', label='NN')
                    # plt.bar(
                    #     list(range(histograms_1.shape[-1])),
                    #     histograms_1[setting_idx, species_idx, 1],
                    #     label='gillespy', alpha=0.7
                    # )
                    # plt.bar(
                    #     list(range(histograms_2.shape[-1])),
                    #     histograms_2[setting_idx, species_idx, 1],
                    #     label='NN', alpha=0.7
                    # )
                    plt.legend()
                    plt.savefig(save_path)
                    plt.close(fig)

        end = time()
        LOGGER.info(f"Took {end - start:.1f} seconds")

    LOGGER.info("All done.")
