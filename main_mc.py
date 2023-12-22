import argparse
import os
import numpy as np
from src.Environment import EnvironmentParams, Environment
from utils import read_config
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf


def eval_logs(event_path):
    pass


def mc(args, params: EnvironmentParams):
    if args.num_agents is not None:
        num_range = [int(i) for i in args.num_agents]
        params.grid_params.num_agents_range = num_range
        env = Environment(params)
        env.agent.load_weights(args.weights)
        env.eval(int(args.samples), show=args.show)
        success_landing = np.array(env.land_ratio,dtype=np.float32)/int(args.samples)
        print('Successful Landing:',success_landing)
        cr = np.array(env.cr,dtype=np.float32)/int(args.samples)
        print('Collection Ratio:',cr)
        print('Collection ratio and landed:',success_landing*cr)

        # print('nfz', env.grid.map_image.start_land_zone)
        # np.savetxt('urban50_start_land_zone.txt', np.c_[env.grid.map_image.start_land_zone], fmt='%d',delimiter='\t')


    return
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to weights')
    parser.add_argument('--config', required=True, help='Config file for agent shaping')
    parser.add_argument('--id', required=False, help='Id for exported files')
    parser.add_argument('--samples', required=True, help='Id for exported files')
    parser.add_argument('--seed', default=43, help="Seed for repeatability")
    parser.add_argument('--show', default=False, help="Show individual plots, allows saving")
    parser.add_argument('--num_agents', default=[5,5], help='Overrides num agents range, argument 12 for range [1,2]')

    args = parser.parse_args()

    if args.seed:
        np.random.seed(int(args.seed))

    params = read_config(args.config)

    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    mc(args, params)
