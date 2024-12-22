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


# def preprocess(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     max_len: int,
#     system_message: str = "You are a helpful assistant."
# ) -> Dict:
#     roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    
#     # Get the special tokens from tokenizer
#     chat_format = {
#         "system": "<|im_start|>system\n{}\n<|im_end|>\n",
#         "user": "<|im_start|>user\n{}\n<|im_end|>\n",
#         "assistant": "<|im_start|>assistant\n{}\n<|im_end|>\n"
#     }

#     # Apply prompt templates
#     input_ids, targets = [], []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != roles["user"]:
#             source = source[1:]

#         input_id, target = [], []
        
#         # Add system message
#         system_text = chat_format["system"].format(system_message)
#         system_tokens = tokenizer.encode(system_text)
#         input_id += system_tokens
#         target += [IGNORE_TOKEN_ID] * len(system_tokens)
        
#         for j, sentence in enumerate(source):
#             role = sentence["from"]
#             text = chat_format[role].format(sentence["value"])
#             tokens = tokenizer.encode(text)
            
#             input_id += tokens
#             if role == "user":
#                 target += [IGNORE_TOKEN_ID] * len(tokens)
#             elif role == "assistant":
#                 target += tokens
#             else:
#                 raise NotImplementedError
                
#         # Truncate and pad
#         input_id = input_id[:max_len]
#         target = target[:max_len]
#         input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
#         target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        
#         input_ids.append(input_id)
#         targets.append(target)

#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     targets = torch.tensor(targets, dtype=torch.long)

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#         attention_mask=input_ids.ne(tokenizer.pad_token_id),
#     )

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
        # super(LazySupervisedDataset, self).__init__()
        # self.tokenizer = tokenizer
        # self.max_len = max_len

        # rank0_print("Formatting inputs...Skip in lazy mode")
        # self.tokenizer = tokenizer
        # self.raw_data = raw_data
        # self.cached_data_dict = {}

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.raw_data = raw_data  # expecting list of dict with {"text": "..."} or similar
        self.cached = {}


    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # if i in self.cached_data_dict:
        #     return self.cached_data_dict[i]

        # ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        # ret = dict(
        #     input_ids=ret["input_ids"][0],
        #     labels=ret["labels"][0],
        #     attention_mask=ret["attention_mask"][0],
        # )
        # self.cached_data_dict[i] = ret

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
    data_args,
    max_len,
) -> Dict:
    """
    Hard-code the loading of each dataset type (RedPajama, Magpie, etc.).
    We'll use `dataset_name` or a similar flag to decide which branch to load.
    Adjust as needed for your real file paths or huggingface load calls.
    """
    # You could pass dataset_name in via data_args or some additional arg:
    dataset_name = os.getenv("DATASET_NAME", "redpajama").lower()
    dataset_name = data_args.dataset_name.lower() if data_args.dataset_name else dataset_name
    # or do: dataset_name = data_args.dataset_name.lower() if you have that field

    # Decide which dataset to load
    if "redpajama" in dataset_name:
        # Example: load from Hugging Face or local. 
        # Or if you already have local JSON: 
        # raw_train_data = json.load(open(data_args.data_path, "r"))
        # 
        # For demonstration, let's pretend we load from a HF dataset:
        from datasets import load_dataset
        raw = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train[:1000]")
        # Convert them into Python list of dict with "text" key
        train_data = [{"text": ex["text"]} for ex in raw]

        # Similarly for eval if data_args.eval_data_path is set:
        eval_data = None
        if data_args.eval_data_path:
            # e.g. load a small eval subset (this is just an example)
            raw_eval = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train[1000:1100]")
            eval_data = [{"text": ex["text"]} for ex in raw_eval]

    elif "magpie" in dataset_name:
        from datasets import load_dataset
        raw = load_dataset("Magpie-Align/Magpie-Air-300K-Filtered", split="train[:2000]")
        # Transform the 'conversations' into plain text if needed
        train_data = []
        for ex in raw:
            # Example: combine all conversation turns into a single "text"
            text_concat = ""
            for c in ex["conversations"]:
                # c["from"] = "human"/"gpt", c["value"] = ...
                text_concat += c["from"].upper() + ": " + c["value"] + "\n"
            train_data.append({"text": text_concat.strip()})

        eval_data = None
        if data_args.eval_data_path:
            # Another small subset for eval, or load a local file, etc.
            eval_data = ...  # same logic, if desired

    else:
        # Default fallback: assume local JSON file with {"text": "..."}
        train_data = json.load(open(data_args.data_path, "r"))
        eval_data = None
        if data_args.eval_data_path:
            eval_data = json.load(open(data_args.eval_data_path, "r"))

    # Now wrap in your LazySupervisedDataset (or SupervisedDataset) 
    # as you currently do for plain-text.
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    train_dataset = dataset_cls(train_data, tokenizer, max_len)

    eval_dataset = None
    if eval_data is not None:
        eval_dataset = dataset_cls(eval_data, tokenizer, max_len)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset
    }





# def make_supervised_data_module(
#     tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
# ) -> Dict:
#     # """Make dataset and collator for supervised fine-tuning."""
#     # dataset_cls = (
#     #     LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
#     # )
#     # rank0_print("Loading data...")

#     # train_json = json.load(open(data_args.data_path, "r"))
#     # train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

#     # if data_args.eval_data_path:
#     #     eval_json = json.load(open(data_args.eval_data_path, "r"))
#     #     eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
#     # else:
#     #     eval_dataset = None

#     # return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
    
#     # Modified to assume data_path points to a plain-text JSON lines or similar.
#     # Example of raw_data: [{"text": "some text..."}, {"text": "another line..."} ...]
#     dataset_cls = (LazySupervisedDataset
#                 if data_args.lazy_preprocess
#                 else SupervisedDataset)
#     raw_train_data = json.load(open(data_args.data_path, "r"))
#     train_dataset = dataset_cls(raw_train_data, tokenizer, max_len)

#     eval_dataset = None
#     if data_args.eval_data_path:
#         raw_eval_data = json.load(open(data_args.eval_data_path, "r"))
#         eval_dataset = dataset_cls(raw_eval_data, tokenizer, max_len)

#     return {
#         "train_dataset": train_dataset,
#         "eval_dataset": eval_dataset
#     }



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

    wandb.init(project=training_args.wandb_project, name=training_args.wandb_run_name)

    # If you didn't set `report_to=["wandb"]` in your TrainingArguments definition,
    # you can set it here at runtime:
    training_args.report_to = ["wandb"]

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
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
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
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Qwen2 uses eos_token_id instead of eod_id

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
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)
    ### ADDED FOR WANDB ###
    # Finish the wandb run to ensure logs are uploaded
    wandb.finish()


if __name__ == "__main__":
    train()
