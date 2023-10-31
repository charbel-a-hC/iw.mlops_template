import json
import wandb
from TFRunner import TFRunner
import os
import sys
import argparse

from mlops_utils.push_metrics import populate_push_metrics_parser

def populate_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--project",
        dest="project",
        type=str,
        help="Weights and Biases Project name"
    )

    parser.add_argument(
        "--entity",
        dest="entity",
        type=str,
        default="idealworks-ml",
        help="Weights and Biases Entity name. Defaults to idealworks entity"
    )

    parser.add_argument(
        "--config",
        dest="config",
        default=f"${os.getcwd()}.config.json",
        type=str,
        help="Path to sweep and project config file. Defaults to any .config.json file in current directory"
    )
    return parser
    
def main(args: argparse.Namespace): 
    
    with open(args.config, "r") as f:
        config = json.load(f)

    sweep_dict = config["sweeps_params"]
    sweep_id = wandb.sweep(sweep_dict, project=args.project, entity=args.entity)

    sample = TFRunner(sweep_id=sweep_id, wandb_project=args.project, 
        wandb_entity=args.entity, wandb_tags=["sample", "mlops"], push_args=args)
    wandb.agent(sweep_id, project=args.project, entity=args.entity, function=sample.fit)
    sample.update_best_model()

if __name__ == "__main__":
    parser = populate_parser(argparse.ArgumentParser(description="Sample Training"))
    parser = populate_push_metrics_parser(parser)
    args, _ = parser.parse_known_args()

    # Exit and print args if not arguments given in command
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    main(args)