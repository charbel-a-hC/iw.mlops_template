import argparse
import wandb
from mlops_utils.parse_readme import parse_readme
from mlops_utils.logger import get_logger

def populate_push_metrics_parser(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--readme_path",
        dest="readme_path",
        type=str,
        default="README.md",
        help="README/Doc path to get WandB URL")
    
    parser.add_argument(
        "--push_metric",
        type=str,
        help="Metric name as specified in WandB")
    
    parser.add_argument(
        "--push_threshold",
        dest="push_threshold",
        type=float,
        help="Metric (as specified in WandB) threhsold"
    )
    
    return parser

def get_push_metric(args: argparse.ArgumentParser):
    entity_name, project_name, run_id = parse_readme(args.readme_path)
    metric = None
    if entity_name:
        wandb.init(project=project_name, entity=entity_name, id=run_id, resume=True)
        metric = wandb.summary[args.push_metric]
    return metric, run_id

def push_metrics(args: argparse.ArgumentParser):
    logger = get_logger("push_metrics")
    update_url = False
    metric, run_id = get_push_metric(args)

    if metric >= args.push_threshold:
        logger.info(f"{args.push_metric} > metric threshold. PUSHING CHANGES WILL OVERRIDE URL @ RUN: {run_id}")
        update_url = True
    else:
        logger.info(f"{args.push_metric} < metric threshold. NO CHANGES WILL BE APPLIED")

    return update_url

if __name__ == "__main__":

    parser = populate_push_metrics_parser(argparse.ArgumentParser(description="Push Metrics"))
    args, _ = parser.parse_args()

    push_metrics(args)