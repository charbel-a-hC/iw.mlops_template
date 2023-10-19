# import your main training function

import json
from functools import partial
import wandb
from TFRunner import TFRunner


with open("sample_config.json", "r") as f:
        config = json.load(f)

sweep_dict = config["sweeps_params"]
sweep_id = wandb.sweep(sweep_dict, project="sample-mlops", entity="idealworks-ml")

sample = TFRunner(sweep_id=sweep_id, wandb_project="sample-mlops", 
                    wandb_entity="idealworks-ml", wandb_tags=["sample", "mlops"])

wandb.agent(sweep_id, project="sample-mlops", entity="idealworks-ml", function=sample.fit)