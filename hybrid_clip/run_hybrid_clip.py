#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training a CLIP like dual encoder models using text and vision encoders in the library.

The script can be used to train CLIP like models for languages other than english by using
a text encoder pre-trained in the desired language. Currently this script support the following vision
and text models:
Vision models: ViT(https://huggingface.co/models?filter=vit), CLIP (https://huggingface.co/models?filter=clip)
Text models: BERT, ROBERTa (https://huggingface.co/models?filter=masked-lm)
"""

import json
import logging
import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv("../.env")

from comet_ml import Experiment


import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import (
    CenterCrop,
    ConvertImageDtype,
    Normalize,
    Resize,
    ColorJitter,
    RandomHorizontalFlip,
    RandomRotation,
    RandomCrop,
    RandomAffine,
    RandomPerspective,
    RandomAutocontrast,
    RandomEqualize,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import transformers
from flax import jax_utils
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, shard, shard_prng_key
from modeling_hybrid_clip import FlaxHybridCLIP
from configuration_hybrid_clip import HybridCLIPConfig
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)
from numpy.random import default_rng


logger = logging.getLogger(__name__)

# Cache the result
has_tensorboard = is_tensorboard_available()
if has_tensorboard:
    try:
        from flax.metrics.tensorboard import SummaryWriter
    except ImportError as ie:
        has_tensorboard = False
        print(
            f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
        )

else:
    print(
        "Unable to display metrics through TensorBoard because the package is not installed: "
        "Please run pip install tensorboard to enable."
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    text_model_name_or_path: str = field(
        metadata={
            "help": "The text model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    vision_model_name_or_path: str = field(
        metadata={
            "help": "The vision model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    from_pt: bool = field(
        default=True,
        metadata={
            "help": "whether to load the text and vision model using PyTorch checkpoints."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory containing input files."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a jsonlines file)."},
    )
    max_seq_length: Optional[int] = field(
        default=72,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension == "json", "`train_file` should be a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension == "json", "`validation_file` should be a json file."


# We use torchvision for faster image pre-processing.
# We need to ensure faster processing speed as it can become a bottleneck on TPU
class Transform(torch.nn.Module):
    def __init__(self, image_size, augment=False):
        super().__init__()
        if not augment:
            self.transforms = torch.nn.Sequential(
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
                ConvertImageDtype(torch.float),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            )
        else:
            self.transforms = torch.nn.Sequential(
                Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                # CenterCrop(image_size),
                RandomCrop([image_size], pad_if_needed=True, padding_mode="edge"),
                ColorJitter(),
                RandomHorizontalFlip(),
                # RandomRotation(15, interpolation=InterpolationMode.BILINEAR, fill=128),
                RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                    shear=(-15, 15, -15, 15),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=127,
                ),
                RandomPerspective(
                    distortion_scale=0.3,
                    p=0.3,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=127,
                ),
                RandomAutocontrast(p=0.3),
                RandomEqualize(p=0.3),
                ConvertImageDtype(torch.float),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


class ImageTextDataset(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths and associated captions.
            The expected format is jsonlines where each line is a json object containing to keys.
            `image_path`: The path to the image.
            `captions`: An `array` of captions.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        captions_per_image=-1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        seed=42,
    ):
        super().__init__(root, transforms, transform, target_transform)
        with open(file_path, "r") as f:
            examples = [json.loads(line) for line in f.readlines()]

        self.rand_generator = default_rng(seed)

        self.captions = []
        self.image_paths = []

        for example in examples:
            if captions_per_image <= -1:
                self.captions.append(example["captions"])
            elif captions_per_image > 0:
                self.captions.append(example["captions"][:captions_per_image])
            else:
                raise ValueError("captions per image cannot be zero")

            self.image_paths.append(example["image_path"])

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        im = read_image(path, mode=ImageReadMode.RGB)
        return im

    def _load_target(self, idx):
        return self.rand_generator.choice(self.captions[idx])
        # if len(self.captions[idx]) > 1:
        #     caption_idx = np.random.randint(0, len(self.captions[idx]))
        # else:
        #     caption_idx = 0
        # return self.captions[idx][caption_idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(
            dropout_rng=shard_prng_key(self.dropout_rng)
        )


def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def log_on_comet(experiment, train_metrics, eval_metrics, train_time, step):
    assert experiment is not None
    experiment.log_metric("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            experiment.log_metric(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        experiment.log_metric(f"eval_{metric_name}", value, step)


def setup_comet():
    logger.info("Comet ML logging requested")
    try:

        if "COMET_API_KEY" in os.environ:
            # Create an experiment with your api key
            experiment = Experiment(
                api_key=os.environ["COMET_API_KEY"],
                project_name="clip-italian",
                workspace="g8a9",
                log_code=True,
                log_graph=False,
            )
            experiment.add_tag("training")
            return experiment
        else:
            logger.info("Can't find COMET_API_KEY env variable, disabling Comet")
            return None

    except:
        logger.info("Something went wrong initializing Comet")
        return None


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    linear=False,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    if linear:
        warmup_fn = optax.linear_schedule(
            init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
        )
        decay_fn = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0,
            transition_steps=num_train_steps - num_warmup_steps,
        )
    else:
        warmup_fn = optax.linear_schedule(
            init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
        )
        decay_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=num_train_steps - num_warmup_steps,
            alpha=0.0,
        )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    parser.add_argument("--log_comet", action="store_true")
    parser.add_argument("--eval_when", type=int, default=1)
    parser.add_argument("--run_from_checkpoint", type=str, default=None)
    parser.add_argument("--margin", type=int, default=None)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            args,
        ) = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    elif model_args.text_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.log_comet:
        comet_exp = setup_comet()

    eval_when = args.eval_when

    if args.run_from_checkpoint is not None:
        with open(f"{args.run_from_checkpoint}/config.json", "r") as fp:
            config_dict = json.load(fp)
        config_dict["vision_config"]["model_type"] = "clip"
        config = HybridCLIPConfig(**config_dict)
        model = FlaxHybridCLIP.from_pretrained(
            args.run_from_checkpoint,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            config=config,
        )
    else:

        model = FlaxHybridCLIP.from_text_vision_pretrained(
            model_args.text_model_name_or_path,
            model_args.vision_model_name_or_path,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            text_from_pt=model_args.from_pt,
            vision_from_pt=model_args.from_pt,
        )
    config = model.config
    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize torchvision transforms and jit them for faster processing
    train_preprocess = Transform(config.vision_config.image_size, augment=True)
    train_preprocess = torch.jit.script(train_preprocess)

    val_preprocess = Transform(config.vision_config.image_size)
    val_preprocess = torch.jit.script(val_preprocess)

    # Initialize the image-text dataset
    train_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.train_file,
        captions_per_image=-1,
        transform=train_preprocess,
        seed=training_args.seed,
    )

    eval_dataset = ImageTextDataset(
        data_args.data_dir,
        data_args.validation_file,
        captions_per_image=-1,
        transform=val_preprocess,
        seed=training_args.seed,
    )

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = (
        int(training_args.per_device_train_batch_size) * jax.device_count()
    )
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Use collate function to tokenizer the text and convert the processed images to numpy
    def collate_fn(examples):
        pixel_values = (
            torch.stack([example[0] for example in examples])
            .permute(0, 2, 3, 1)
            .numpy()
        )
        captions = [example[1] for example in examples]
        inputs = tokenizer(
            captions,
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        batch = {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        return batch

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=data_args.preprocessing_num_workers,
        persistent_workers=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Enable tensorboard only on the master node
    if has_tensorboard and jax.process_index() == 0:
        summary_writer = SummaryWriter(
            log_dir=Path(training_args.output_dir).joinpath("logs").as_posix()
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Create learning rate schedule
    if training_args.warmup_steps:
        warmup_steps = training_args.warmup_steps
    elif training_args.warmup_ratio:
        warmup_steps = int(training_args.warmup_ratio * total_train_steps)
    else:
        raise RuntimeError(
            "You have to specify either the warmup_steps or warmup_ratio CLI parameter"
        )

    decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        training_args.num_train_epochs,
        warmup_steps,
        training_args.learning_rate,
        linear=False,  # set False to activate cosine annealing
    )

    # create adam optimizer
    #     optimizer = optax.adamw(
    #         learning_rate=decay_lr_schedule_fn,
    #         b1=training_args.adam_beta1,
    #         b2=training_args.adam_beta2,
    #         eps=training_args.adam_epsilon,
    #         weight_decay=training_args.weight_decay,
    #     )

    optimizer = optax.chain(
        optax.adaptive_grad_clip(0.01, eps=0.001),
        optax.scale_by_belief(),
        optax.scale_by_schedule(decay_lr_schedule_fn),
        optax.scale(-1.0),
    )

    # Setup train state
    state = TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer,
        dropout_rng=dropout_rng,
    )

    def cross_entropy(logits, axis):
        logprobs = jax.nn.log_softmax(logits, axis=axis)
        nll = jnp.diag(logprobs)
        ce = -jnp.mean(nll)
        return ce

    def clip_loss(similarity):

        jax_array = jax.numpy.full(similarity.shape, 0.2)

        diag_elements = jnp.diag_indices_from(jax_array)
        margin_matrix = jax_array.at[diag_elements].set(0)

        new_similarity = similarity + margin_matrix

        loss = (
            cross_entropy(new_similarity, axis=0) + cross_entropy(new_similarity, axis=1)
        ) / 2

        return loss

    # Define gradient update step fn
    def train_step(state, batch):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            logits = state.apply_fn(
                **batch, params=params, dropout_rng=dropout_rng, train=True
            )[0]
            loss = clip_loss(logits)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {
            "loss": loss,
            "learning_rate": decay_lr_schedule_fn(state.step),
        }
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics

    # Define eval fn
    def eval_step(params, batch):
        logits = model(**batch, params=params, train=False)[0]
        loss = clip_loss(logits)

        # summarize metrics
        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {train_batch_size}"
    )
    logger.info(f"  Total optimization steps = {total_train_steps}")
    logger.info(f"  Total warmup steps = {warmup_steps}")

    train_time = 0
    # Create sampling rng
    rng, input_rng = jax.random.split(rng)

    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []

        steps_per_epoch = len(train_dataset) // train_batch_size
        train_step_progress_bar = tqdm(
            total=steps_per_epoch, desc="Training...", position=1, leave=False
        )
        # train
        for batch in train_loader:
            batch = shard(batch)
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

        train_time += time.time() - train_start

        train_metric = unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})"
        )

        # ======================== Evaluating ==============================

        if epoch % eval_when == 0:

            eval_metrics = []
            eval_steps = len(eval_dataset) // eval_batch_size
            eval_step_progress_bar = tqdm(
                total=eval_steps, desc="Evaluating...", position=2, leave=False
            )
            for batch in eval_loader:
                # Model forward
                batch = shard(batch)
                metrics = p_eval_step(state.params, batch)
                eval_metrics.append(metrics)

                eval_step_progress_bar.update(1)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)

            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

            # Print metrics and update progress bar
            eval_step_progress_bar.close()
            desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']})"
            epochs.write(desc)
            epochs.desc = desc

        # Save metrics
        if has_tensorboard and jax.process_index() == 0:
            cur_step = epoch * (len(train_dataset) // train_batch_size)
            write_metric(
                summary_writer, train_metrics, eval_metrics, train_time, cur_step
            )
        if args.log_comet and comet_exp is not None and jax.process_index() == 0:
            cur_step = epoch * (len(train_dataset) // train_batch_size)
            log_on_comet(comet_exp, train_metrics, eval_metrics, train_time, cur_step)

        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(unreplicate(state.params))
            model.save_pretrained(
                training_args.output_dir + f"/{epoch+1}/",
                params=params,
                push_to_hub=training_args.push_to_hub,
                commit_message=f"Saving weights and logs of epoch {epoch+1}",
            )


if __name__ == "__main__":
    main()
