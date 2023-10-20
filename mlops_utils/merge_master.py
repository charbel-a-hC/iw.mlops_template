import wandb
import argparse
import wandb
from parse_readme import parse_readme

parser = argparse.ArgumentParser()
parser.add_argument("--readme_path", type=str)
args = parser.parse_args()

entity_name, project_name, run_id = parse_readme(args.readme_path)

api = wandb.Api()

old_run = api.runs(path=entity_name + "/" + project_name, filters={"tags": "current"})[
    0
].id

with wandb.init(
    project=project_name, entity=entity_name, id=old_run, resume=True
) as run_old:
    old_metric = run_old.summary["testing/avg_iou"]

with wandb.init(
    project=project_name, entity=entity_name, id=run_id, resume=True
) as run:
    metric = run.summary["testing/avg_iou"]

assert metric > old_metric

run = api.run(entity_name + "/" + project_name + "/" + run_id)
# run.tags.append("current")
run.update()

run_old = api.run(entity_name + "/" + project_name + "/" + old_run)
# run_old.tags.remove("current")
run_old.update()
