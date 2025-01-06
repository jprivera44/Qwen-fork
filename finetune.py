# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import transformers
from transformers import Trainer, GPTQConfig
import transformers.integrations.deepspeed as deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType

import wandb

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False

    dataset_name: str = field(
        default="default name",
        metadata={"help": "dataset name"}
    )

    subset_size: int = field(
        default=1000,
        metadata={"help": "Number of examples to use for training. -1 means use all available data."}
    )

    rp_subset_size: int = field(
        default=10000,
        metadata={"help": "Number of examples for RedPajama. -1 = use full data."}
    )

    magpie_subset_size: int = field(
        default=10000,
        metadata={"help": "Number of examples for Magpie. -1 = use full data."}
    )
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    ### ADDED FOR WANDB ###
    # Let HF know we want to report logs to wandb.
    # If you prefer, you can also do this at runtime using:
    #   training_args.report_to = ["wandb"]
    # in the `train()` function.
    

    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["wandb"], 
        metadata={"help": "List of integrations to report the results and logs to."}
    )

    wandb_project: str = field(
        default="compression_pythia_scaling_sweep",
        metadata={"help": "Wandb project name."}
    )

    wandb_run_name: str = field(
        default="default name",
        metadata={"help": "Wandb run name."}
    )

    gradient_clipping: float = 1.0




@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)



#############################
# New plain-text preprocess #
#############################
def preprocess_plain_text(
    texts: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int
) -> Dict[str, torch.Tensor]:
    # Tokenize plain text examples
    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    # Use input_ids as labels for causal LM
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings




class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.raw_data = raw_data  # expecting list of dict with {"text": "..."} or similar
        self.cached = {}


    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        # return ret
        if i in self.cached:
            return self.cached[i]

        text_example = self.raw_data[i]["text"]  # adapt key if needed
        data_dict = preprocess_plain_text([text_example], self.tokenizer, self.max_len)
        item = {
            "input_ids": data_dict["input_ids"][0],
            "attention_mask": data_dict["attention_mask"][0],
            "labels": data_dict["labels"][0],
        }
        self.cached[i] = item
        return item
    

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    max_len: int,
) -> Dict:
    dataset_name = (data_args.dataset_name or os.getenv("DATASET_NAME", "redpajama")).lower()

    if "redpajama" in dataset_name:
        from datasets import load_dataset, load_from_disk

        if data_args.data_path and os.path.exists(data_args.data_path):
            full_dataset = load_from_disk(data_args.data_path)
        else:
            full_dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")

        full_dataset = full_dataset.shuffle(seed=42)

        # Grab the subset size for RedPajama
        rp_size = data_args.rp_subset_size
        # -1 means "use full" -> interpret how you want (here we just won't slice)
        if rp_size == -1:
            rp_size = len(full_dataset)

        total_size = min(rp_size + 1000, len(full_dataset))
        train_size = min(rp_size, total_size - 1000)
        eval_size = min(1000, total_size - train_size)

        print(f"[RedPajama] rp_subset_size param: {data_args.rp_subset_size}")
        print(f"[RedPajama] total_size: {total_size}, train_size: {train_size}, eval_size: {eval_size}")

        raw_train = full_dataset.select(range(train_size))
        raw_eval = full_dataset.select(range(train_size, train_size + eval_size))

        train_data = [{"text": ex["text"]} for ex in raw_train]
        eval_data = [{"text": ex["text"]} for ex in raw_eval]

    elif "magpie" in dataset_name:
        from datasets import load_dataset
        full_dataset = load_dataset("Magpie-Align/Magpie-Air-300K-Filtered", split="train")
        full_dataset = full_dataset.shuffle(seed=42)

        # Grab the subset size for Magpie
        mg_size = data_args.magpie_subset_size
        if mg_size == -1:
            mg_size = len(full_dataset)

        total_size = min(mg_size + 1000, len(full_dataset))
        train_size = min(mg_size, total_size - 1000)
        eval_size = min(1000, total_size - train_size)

        print(f"[Magpie]   magpie_subset_size param: {data_args.magpie_subset_size}")
        print(f"[Magpie]   total_size: {total_size}, train_size: {train_size}, eval_size: {eval_size}")

        raw_train = full_dataset.select(range(train_size))
        raw_eval = full_dataset.select(range(train_size, train_size + eval_size))

        train_data = []
        for ex in raw_train:
            text_concat = ""
            for c in ex["conversations"]:
                text_concat += f"{c['from'].upper()}: {c['value']}\n"
            train_data.append({"text": text_concat.strip()})

        eval_data = []
        for ex in raw_eval:
            text_concat = ""
            for c in ex["conversations"]:
                text_concat += f"{c['from'].upper()}: {c['value']}\n"
            eval_data.append({"text": text_concat.strip()})

    else:
        # Some fallback logic
        train_data = json.load(open(data_args.data_path, "r"))
        eval_data = None
        if data_args.eval_data_path:
            eval_data = json.load(open(data_args.eval_data_path, "r"))

    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset

    train_dataset = dataset_cls(train_data, tokenizer, max_len)
    eval_dataset = dataset_cls(eval_data, tokenizer, max_len) if eval_data is not None else None

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset
    }









def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # Initialize W&B
    wandb.init(project=training_args.wandb_project, name=training_args.wandb_run_name)
    training_args.report_to = ["wandb"]  # Ensure we log to W&B

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are incompatible with QLoRA.")

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on a base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Load config + model
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(bits=4, disable_exllama=True)
            if training_args.use_lora and lora_args.q_lora
            else None,
        **model_load_kwargs,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If using LoRA
    if training_args.use_lora:
        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    ################################################################
    # 1) First finetuning stage: Magpie
    ################################################################

    # 1) Create the Trainer once
    data_args.dataset_name = "magpie"

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args, 
        max_len=training_args.model_max_length,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # 2) Train on the first dataset
    print("=== Starting first fine-tuning pass on Magpie ===")
    trainer.train()
    trainer.save_state()

    # 3) Change the dataset to RedPajama
    data_args.dataset_name = "redpajama"
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, 
        data_args=data_args, 
        max_len=training_args.model_max_length,
    )

    # Manually update trainer's datasets
    trainer.train_dataset = data_module["train_dataset"]
    trainer.eval_dataset = data_module["eval_dataset"]

    # 4) Train on the new dataset
    print("=== Starting second fine-tuning pass on RedPajama ===")
    trainer.train()
    trainer.save_state()

    ################################################################
    # 3) Save final model
    ################################################################
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias
    )

    # Finish W&B
    wandb.finish()
    print("=== Done! Model fine-tuned on Magpie, then RedPajama ===")






if __name__ == "__main__":
    train()
