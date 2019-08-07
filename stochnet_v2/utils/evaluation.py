import numpy as np
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.static_classes.model import StochNet


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
        n_steps,
        n_traces_per_setting,
        path_to_save_generated_data=None,
):
    nn = StochNet(
        nb_past_timesteps=nb_past_timesteps,
        nb_features=nb_features,
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        mode='inference'
    )

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)
    histogram_data = np.load(dataset_explorer.histogram_dataset_fp)
    initial_settings = histogram_data[:, 0, 0:nb_past_timesteps, -nb_features:]

    print("Start generating NN traces")
    cnt = 0
    while True:
        if cnt > 10:
            print(f"Failed to generate NN traces after {cnt} attempts...")
            break
        try:
            traces = nn.generate_traces(
                initial_settings,
                n_steps=n_steps,
                n_traces=n_traces_per_setting,
                curr_state_rescaled=False,
                round_result=True,
                add_timesteps=True,
            )
            print(f"Done. generated data shape: {traces.shape}")
            if path_to_save_generated_data:
                np.save(path_to_save_generated_data, traces)
            return traces
        except:
            print("Oops... trying again")
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
                # normed=True,
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
    distance = np.sum(distance, axis=(0, -1)) / (n_settings)

    return distance


def get_histogram_distance(
        data_1,
        data_2,
        time_lag,
        n_bins=100,
        target_species_idxs=None,
        histogram_bounds=None,
        with_timestamps=True,
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

    distance = _histogram_distance(histograms_1, histograms_2)
    return distance, histograms_1, histograms_2
