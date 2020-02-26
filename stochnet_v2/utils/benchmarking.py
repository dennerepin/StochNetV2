import numpy as np
from time import time
from stochnet_v2.utils.util import generate_gillespy_traces


def benchmark(m, nn, timestep, n_settings, traj_per_setting, n_steps):
    initial_settings = m.get_initial_settings(n_settings)

    start = time()
    gillespy_traces = generate_gillespy_traces(
        settings=initial_settings,
        n_steps=n_steps,
        timestep=timestep,
        gillespy_model=m,
        traj_per_setting=traj_per_setting,
    )
    gillespy_time = time() - start

    start = time()
    nn_traces = nn.generate_traces(
        initial_settings[:, np.newaxis, :],
        n_steps=n_steps,
        n_traces=traj_per_setting,
        curr_state_rescaled=False,
        scale_back_result=True,
        round_result=True,
        add_timestamps=True
    )
    nn_time = time() - start

    return gillespy_time, nn_time


def benchmark_ssa(m, timestep, n_settings, traj_per_setting, n_steps):
    initial_settings = m.get_initial_settings(n_settings)
    start = time()
    gillespy_traces = generate_gillespy_traces(
        settings=initial_settings,
        n_steps=n_steps,
        timestep=timestep,
        gillespy_model=m,
        traj_per_setting=traj_per_setting,
    )
    gillespy_time = time() - start
    return gillespy_time


def benchmark_nn(m, nn, n_settings, traj_per_setting, n_steps):
    initial_settings = m.get_initial_settings(n_settings)
    start = time()
    nn_traces = nn.generate_traces(
        initial_settings[:, np.newaxis, :],
        n_steps=n_steps,
        n_traces=traj_per_setting,
        curr_state_rescaled=False,
        scale_back_result=True,
        round_result=True,
        add_timestamps=True
    )
    nn_time = time() - start
    return nn_time
