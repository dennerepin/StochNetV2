import argparse
import os
from time import time

from stochnet_v2.static_classes.model import StochNet
from stochnet_v2.static_classes.trainer import Trainer
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.utils.util import str_to_bool


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, required=True)
    parser.add_argument('--timestep', type=float, required=True)
    parser.add_argument('--dataset_id', type=int, required=True)
    parser.add_argument('--model_id', type=int, required=True)
    parser.add_argument('--nb_features', type=int, required=True)
    parser.add_argument('--nb_past_timesteps', type=int, required=True)
    parser.add_argument('--nb_randomized_params', type=int, required=True)

    parser.add_argument('--body_config_path', type=str, required=True)
    parser.add_argument('--mixture_config_path', type=str, required=True)

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--add_noise', type=str, default='false', choices=['true', 'false', 'True', 'False'])
    parser.add_argument('--stddev', type=float, default=0.01)
    parser.add_argument('--dataset_kind', type=str, default='hdf5')
    
    args = parser.parse_args()

    project_folder = args.project_folder
    timestep = args.timestep
    dataset_id = args.dataset_id
    model_id = args.model_id
    nb_features = args.nb_features
    nb_past_timesteps = args.nb_past_timesteps
    nb_randomized_params = args.nb_randomized_params

    body_config_path = args.body_config_path
    mixture_config_path = args.mixture_config_path

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    add_noise = str_to_bool(args.add_noise)
    stddev = args.stddev

    dataset_kind = args.dataset_kind
    learning_strategy = None
    ckpt_path = None

    project_explorer = ProjectFileExplorer(project_folder)
    dataset_explorer = project_explorer.get_dataset_file_explorer(timestep, dataset_id)
    model_explorer = project_explorer.get_model_file_explorer(timestep, model_id)

    start = time()

    nn = StochNet(
        nb_past_timesteps=nb_past_timesteps,
        nb_features=nb_features,
        nb_randomized_params=nb_randomized_params,
        project_folder=project_folder,
        timestep=timestep,
        dataset_id=dataset_id,
        model_id=model_id,
        body_config_path=body_config_path,
        mixture_config_path=mixture_config_path
    )

    best_ckpt_path = Trainer().train(
        nn,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_strategy=learning_strategy,
        ckpt_path=ckpt_path,
        dataset_kind=dataset_kind,
        add_noise=add_noise,
        stddev=stddev,
    )

    end = time()
    execution_time = end - start
    msg = f"\n\nTraining model {model_id} on dataset {dataset_id} took {execution_time // 60} minutes.\n" \
          f"\tn_epochs={n_epochs} \n" \
          f"\tbatch_size={batch_size} \n" \
          f"\tmodel restored from {best_ckpt_path} saved as {model_explorer.frozen_graph_fp}\n"

    with open(dataset_explorer.log_fp, 'a') as f:
        f.write(msg)

    with open(model_explorer.log_fp, 'a') as f:
        f.write(msg)


if __name__ == "__main__":
    main()


"""
python stochnet_v2/scripts/train_static.py \
    --project_folder='/home/dn/DATA/SIR' \
    --timestep=0.5 \
    --dataset_id=3 \
    --model_id=3001 \
    --nb_features=3 \
    --nb_past_timesteps=1 \
    --nb_randomized_params=2 \
    --body_config_path='/home/dn/DATA/SIR/body_config.json' \
    --mixture_config_path='/home/dn/DATA/SIR/mixture_config.json' \
    --n_epochs=5 \
    --batch_size=5 \
    --add_noise=True \
    --stddev=0.01
"""
