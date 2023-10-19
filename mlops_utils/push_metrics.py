import argparse
import wandb
from utils.parse_readme import parse_readme

parser = argparse.ArgumentParser()
parser.add_argument("--readme_path", type=str)
parser.add_argument("--metric", type=float, default=0.6)
args = parser.parse_args()

entity_name, project_name, run_id = parse_readme(args.readme_path)

wandb.init(project=project_name, entity=entity_name, id=run_id, resume=True)

metric = wandb.summary["testing/avg_iou"]

assert metric >= args.metric
