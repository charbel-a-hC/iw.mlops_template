import tensorflow as tf
from dataclasses import dataclass, fields
import keras
import wandb
from typing import List, Any
from model import create_model
from dataset import get_dataset
from mlops_utils.parse_readme import parse_readme, update_readme
from mlops_utils.logger import get_logger

@dataclass
class TFMetrics:
    train_metric: tf.metrics.Metric = None
    val_metric: tf.metrics.Metric = None


class TFRunner:
    logger = get_logger(description="TFRunner")

    def __init__(self, sweep_id, wandb_project, wandb_entity, wandb_tags, **kwargs):
        # Setup WandB parameters
        self.wandb_project: str = wandb_project
        self.wandb_entity: str = wandb_entity
        self.wandb_tags: List[str] = wandb_tags
        self.sweep_id: int = sweep_id

        # Instantiate a loss function.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Instatiate metrics
        self.metrics = TFMetrics(train_metric= tf.metrics.SparseCategoricalCrossentropy(), 
                                    val_metric=tf.metrics.SparseCategoricalCrossentropy())
        
        self.push_args = kwargs["push_args"]

    def fit(self):

        @tf.function
        def train_step(x, y, optimizer):
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss_value = self.loss_fn(y, logits)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            self.metrics.train_metric.update_state(y, logits)

        @tf.function
        def val_step(x, y):
            val_logits = self.model(x, training=False)
            self.metrics.val_metric.update_state(y, val_logits)

        # Instantiate an optimizer.
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)

        run = wandb.init(
            project=self.wandb_project, entity=self.wandb_entity, tags=self.wandb_tags
        )
        self.config = wandb.config

        self.model = create_model(self.config.dense_layers)
        self.train_dataset, self.val_dataset = get_dataset(self.config.batch_size, self.config.train_samples)

        wandb.run.name = (
                f"sample-mlops-batch_size-{self.config.batch_size}-epochs-{self.config.epochs}"
            )
        
        for epoch in range(self.config.epochs):
            # Iterate over the batches of the dataset.
            for x_batch_train, y_batch_train in self.train_dataset:
                train_step(x_batch_train, y_batch_train, optimizer)
                
            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in self.val_dataset:
                val_step(x_batch_val, y_batch_val)

            # metrics handling
            metrics = {}
            
            for field in fields(self.metrics):
                metrics[field.name] = float(getattr(self.metrics, field.name).result())
                getattr(self.metrics, field.name).reset_states()

            out = ""
            for i, (key, value) in enumerate(metrics.items()):
                out += f"{key}: {value:.4f}"
                if not i == (len(metrics) - 1):
                    out += "\t"
                if "train" in key:
                    wandb.log({f"training/{key}": value})
                if "val" in key:
                    wandb.log({f"testing/{key}": value})

            print(f"Epoch: {epoch}/{self.config.epochs} " + out)

    def update_best_model(self):

        wandb_api = wandb.Api()
        sweep = wandb_api.sweep(f"{self.wandb_entity}/{self.wandb_project}/{self.sweep_id}")

        best_run = sweep.best_run()
        curr_metric = best_run.summary[self.push_args.push_metric]

        _, _, prev_run_id = parse_readme(self.push_args.readme_path)

        if prev_run_id:
            prev_run = wandb_api.run(f"{self.wandb_entity}/{self.wandb_project}/{prev_run_id}")    
            prev_metric = prev_run.summary[self.push_args.push_metric]

            if curr_metric < prev_metric:
                TFRunner.logger.info(f"Best current run is better than README run and < metric threshold ({best_run.id}:{curr_metric:.3f} < {prev_run.id}:{prev_metric:.3f}), updating ...")
                update_readme(run_id=best_run.id, entity=self.wandb_entity, project=self.wandb_project, readme_path=self.push_args.readme_path)
            else:
                TFRunner.logger.info(f"README run is better than current best sweep run ({prev_run.id}:{prev_metric:.3f} < {best_run.id}:{curr_metric:.3f})! README will not be updated!")

        else:
            TFRunner.logger.info("No previous run found! Checking for threshold constraint ...")
            if curr_metric < self.push_args.push_threshold:
                TFRunner.logger.info("Best current run is better than metric threshold, updating ...")
                update_readme(run_id=best_run.id, entity=self.wandb_entity, project=self.wandb_project, readme_path=self.push_args.readme_path)

    
