import tensorflow as tf
from dataclasses import dataclass, fields
import keras
import wandb
from typing import List
from model import create_model
from dataset import get_dataset
from ..mlops_utils.parse_readme import update_readme

@dataclass
class TFMetrics:
    train_metric: tf.metrics.Metric = None
    val_metric: tf.metrics.Metric = None


class TFRunner:
    def __init__(self, sweep_id, wandb_project, wandb_entity, wandb_tags):
        # Setup WandB parameters
        self.wandb_project: str = wandb_project
        self.wandb_entity: str = wandb_entity
        self.wandb_tags: List[str] = wandb_tags
        self.sweep_id: int = sweep_id
        
        # Instantiate an optimizer.
        self.optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        # Instantiate a loss function.
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Instatiate metrics
        self.metrics = TFMetrics(train_metric= tf.metrics.SparseCategoricalCrossentropy(), 
                                 val_metric=tf.metrics.SparseCategoricalCrossentropy())

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.metrics.train_metric.update_state(y, logits)

    @tf.function
    def val_step(self, x, y):
        val_logits = self.model(x, training=False)
        self.metrics.val_metric.update_state(y, val_logits)
    
    def fit(self):
        run = wandb.init(
            project=self.wandb_project, entity=self.wandb_entity, tags=self.wandb_tags
        )
        
        self.config = wandb.config

        self.model = create_model(self.config.dense_layers)
        self.train_dataset, self.val_datset = get_dataset(self.config.batch_size, self.config.train_samples)

        wandb.run.name = (
                f"sample-mlops-batch_size-{self.config.batch_size}-epochs-{self.config.epochs}"
            )
    
        for epoch in range(self.config.epochs):
            # Iterate over the batches of the dataset.
            for x_batch_train, y_batch_train in self.train_dataset:
                self.train_step(x_batch_train, y_batch_train)
                
            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in self.val_dataset:
                self.val_step(x_batch_val, y_batch_val)

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
        
        update_readme(run.get_url().split("/")[-1].split()[0], "./README.md")
