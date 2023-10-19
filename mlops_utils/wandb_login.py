import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_api_key", type=str)
args = parser.parse_args()

wandb.login(relogin=True, key=args.wandb_api_key)
