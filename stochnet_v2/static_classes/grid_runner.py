import numpy as np
import shutil
from collections import namedtuple
from copy import deepcopy
from scipy import signal
from time import time

import multiprocessing
import os
from functools import partial
from stochnet_v2.utils.util import maybe_create_dir
# from stochnet_v2.utils.util import generate_gillespy_traces

GridSpec = namedtuple('GridSpec', ['boundaries', 'grid_size'])
ADJACENTS = [(-1, -1), (-1, 0), (-1, 1),
             (1, 1), (1, 0), (1, -1), (0, 1), (0,-1)]


def _single_trace(
        setting,
        id_number,
        gillespy_model,
        params_to_randomize,
        traj_per_setting,
        save_dir,
        prefix,
):
    # start = time()
    nb_randomized_params = len(params_to_randomize)
    if nb_randomized_params > 0:
        params = setting[-nb_randomized_params:]
        setting = setting[:-nb_randomized_params]
        param_dict = dict(zip(params_to_randomize, params))

        gillespy_model.set_species_initial_value(setting)
        gillespy_model.set_parameters(param_dict)

    else:
        gillespy_model.set_species_initial_value(setting)

    traces = gillespy_model.run(number_of_trajectories=traj_per_setting, show_labels=False)
    traces = np.array(traces)
    # elapsed = time() - start
    # print(f'..single trace: shape={traces.shape}, elapsed {elapsed:.2f}')
    _save_simulation_data(traces, save_dir, prefix, id_number)


def _save_simulation_data(
        data,
        dataset_folder,
        prefix,
        id_number,
):
    partial_dataset_filename = str(prefix) + str(id_number) + '.npy'
    partial_dataset_filepath = os.path.join(dataset_folder, partial_dataset_filename)
    np.save(partial_dataset_filepath, data)
    return


def generate_gillespy_traces(
        settings,
        n_steps,
        timestep,
        gillespy_model,
        params_to_randomize,
        save_dir='/tmp/model_trajectories/',
        traj_per_setting=1,
        prefix='partial_',
):
    # start = time()
    shutil.rmtree(save_dir, ignore_errors=True)
    maybe_create_dir(save_dir)

    nb_settings = len(settings)
    endtime = n_steps * timestep
    nb_of_steps = int(endtime // timestep) + 1

    gillespy_model.timespan(np.linspace(0, endtime, nb_of_steps))

    count = multiprocessing.cpu_count() * 3 // 4
    pool = multiprocessing.Pool(
        processes=count
    )

    task = partial(
        _single_trace,
        gillespy_model=gillespy_model,
        params_to_randomize=params_to_randomize,
        traj_per_setting=traj_per_setting,
        save_dir=save_dir,
        prefix=prefix,
    )

    kwargs = [(settings[n], n) for n in range(nb_settings)]

    pool.starmap(task, kwargs)
    pool.close()

    for i in range(nb_settings):
        filename = str(prefix) + str(i) + '.npy'
        partial_dataset_filepath = os.path.join(
            save_dir,
            filename
        )
        with open(partial_dataset_filepath, 'rb') as f:
            partial_dataset = np.load(f)
        if i == 0:
            simulated_traces = partial_dataset[np.newaxis, ...]
        else:
            simulated_traces = np.concatenate(
                (simulated_traces, partial_dataset[np.newaxis, ...]),
                axis=0,
            )
        os.remove(partial_dataset_filepath)

    shutil.rmtree(save_dir, ignore_errors=True)
    # elapsed = time() - start
    # print(f'..single trace: shape={simulated_traces.shape}, elapsed {elapsed:.2f}')
    return simulated_traces


class Model:
    """Proxy for CRN model with the same simulation interface as StochNet model."""
    def __init__(
            self,
            crn_model,
            params_to_randomize,
    ):
        self.model = deepcopy(crn_model)
        # self.model.timespan(self.model.tspan[:2])
        self.params_to_randomize = params_to_randomize
        self.nb_features = self.model.get_n_species()
        self.nb_randomized_params = len(self.params_to_randomize)
        self.timestep = self.model.tspan[1] - self.model.tspan[0]

    def next_state(self, state, *args, **kwargs):
        model_next_state = generate_gillespy_traces(
            settings=np.squeeze(state, 1),
            n_steps=1,
            timestep=self.timestep,
            gillespy_model=self.model,
            params_to_randomize=self.params_to_randomize,
            traj_per_setting=1,
        )
        model_next_state = model_next_state[:, :, -1:, 1:]  # [n_settings, n_samples, n_steps=1, n_features]
        model_next_state = np.transpose(model_next_state, (1, 0, 2, 3))  # [n_samples, n_settings, 1, n_features]
        return model_next_state


class GridRunner:

    def __init__(
            self,
            model,  # StochNet or Model
            grid_spec: GridSpec,
            save_dir: str,
            diffusion_sigma=0.5,
            diffusion_kernel_size=3,
    ):
        self.model = model
        self.grid = self.create_grid(grid_spec)
        self.save_dir = save_dir
        self._state = np.zeros(
            self.grid.shape[:-1] + (self.model.nb_features + self.model.nb_randomized_params,)
        )
        if diffusion_kernel_size % 2 == 0:
            raise ValueError('Kernel size should be odd.')
        self.diffusion_kernel = np.expand_dims(
            np.outer(
                signal.gaussian(diffusion_kernel_size, diffusion_sigma),
                signal.gaussian(diffusion_kernel_size, diffusion_sigma)),
            -1)

    @property
    def state(self):
        return self._state.copy()

    @staticmethod
    def create_grid(grid_spec):
        x_range = np.linspace(*grid_spec.boundaries[0], grid_spec.grid_size[0])
        y_range = np.linspace(*grid_spec.boundaries[1], grid_spec.grid_size[1])
        xi, yi = np.meshgrid(x_range, y_range)
        xy = (np.array([xi, yi])).T
        return xy

    def set_state(
            self,
            state,
            position=None,
            mode='species',
    ):
        position = position or ...
        if mode == 'all':
            self._state[position] = state
        elif mode == 'species':
            self._state[..., :self.model.nb_features][position] = state
        elif mode == 'params':
            self._state[..., self.model.nb_features:][position] = state
        else:
            raise ValueError('`mode` parameter should be one of {"all", "species", "params"}.')

    def clear_state(self, mode='species'):
        if mode == 'all':
            last_dim = self.model.nb_features + self.model.nb_randomized_params
        elif mode == 'species':
            last_dim = self.model.nb_features
        elif mode == 'params':
            last_dim = self.model.nb_randomized_params
        else:
            raise ValueError('`mode` parameter should be one of {"all", "species", "params"}.')
        self.set_state(
            np.zeros(self.grid.shape[:-1] + (last_dim,)),
            mode=mode,
        )

    def diffusion_step(self, species_idx=None, conservation=True):
        if not self.diffusion_kernel.shape[0] == self.diffusion_kernel.shape[0]:
            raise ValueError('Uneven kernel shape.')
        if species_idx is not None and not species_idx <= self.model.nb_features - 1:
            raise ValueError(f'species_idx={species_idx}, out of range [0, {self.model.nb_features - 1}].')
        k = self.diffusion_kernel.shape[0]

        state = self.state[..., :self.model.nb_features]
        s = state if species_idx is None else state[..., species_idx:species_idx + 1]
        if np.all(s == 0):
            return

        s = np.pad(s, k//2, 'edge')[..., k//2:-(k//2)]
        s = signal.fftconvolve(s, self.diffusion_kernel, mode='valid', axes=None)
        if conservation:
            norm_coeff = 1 / np.sum(self.diffusion_kernel, axis=(0, 1))
            s = s * norm_coeff

        if species_idx is not None:
            state[..., species_idx:species_idx + 1] = s
        else:
            state = s

        self.set_state(state, mode='species')

    def max_propagation_step(self, species_idx=None, alpha=1.0):
        state = self.state[..., :self.model.nb_features]
        s = state if species_idx is None else state[..., species_idx:species_idx + 1]
        if np.all(s == 0):
            return

        s = np.pad(s, 1, 'constant')[..., 1:-1]  # [..., 0]
        # stack = np.stack([np.roll(s, adj, (0, 1)) for adj in ADJACENTS] + [s], axis=0)
        stack = np.stack([np.roll(s, adj, (0, 1)) * alpha for adj in ADJACENTS] + [s], axis=0)
        s = np.max(stack, axis=0)[1:-1, 1:-1]

        if species_idx is not None:
            state[..., species_idx:species_idx + 1] = s
        else:
            state = s

        self.set_state(state, mode='species')

    def model_step(self, threshold=None):
        if threshold is None:
            threshold = np.array([0.999] * self.model.nb_features)
        if not len(threshold) == self.model.nb_features:
            raise ValueError(f'`threshold` should have len={self._state.shape[-1]}.')

        non_zero_idxs = np.where(
            (self._state[..., :self.model.nb_features] > threshold).sum(axis=-1) > 0
        )
        print(f'\t -- non-zero states: {len(non_zero_idxs[0])}')
        if len(non_zero_idxs[0]) == 0:
            return

        non_zero_state = self._state[non_zero_idxs]
        next_state = self.model.next_state(
            non_zero_state[:, np.newaxis, :],  # [n_settings, 1, nb_features]
            curr_state_rescaled=False,
            scale_back_result=True,
            round_result=True,
            n_samples=1,
        )
        next_state = np.squeeze(next_state[0], 1)
        next_state = np.maximum(next_state, 0.)
        self.set_state(next_state, non_zero_idxs, mode='species')

    def run_model(
            self,
            tot_iterations,
            propagation_steps,
            model_steps,
            propagation_mode='mp',  # 'd'
            **propagation_kwargs,
    ):
        if propagation_mode == 'd':
            propagation_fn = lambda: self.diffusion_step(**propagation_kwargs)
        elif propagation_mode == 'mp':
            propagation_fn = lambda: self.max_propagation_step(**propagation_kwargs)
        else:
            raise ValueError(
                f'propagation_mode should be "d" for diffusion or "mp" for max_propagation.')

        states = list()
        states.append(self.state)

        for i in range(tot_iterations):
            print(f'\niter {i}\n')
            outer_start = time()

            for _ in range(propagation_steps):
                if i == 0:
                    continue
                print(f'\tpropagation_step {_}')
                step_start = time()
                propagation_fn()
                elapsed = time() - step_start
                print(f'\t . . . {elapsed:.2f}')
                states.append(self.state)

            for _ in range(model_steps):
                print(f'\tmodel_step {_}')
                step_start = time()
                self.model_step()
                elapsed = time() - step_start
                print(f'\t . . . {elapsed:.2f}')
                states.append(self.state)

            elapsed = time() - outer_start
            print(f' . . . {elapsed:.2f}')

        return states
