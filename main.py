"""main"""
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler
)

from utils import get_bnb_config, get_prompt
from src import Evaluator

def parse_args() -> argparse.Namespace:
    """parse_args"""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--train_file", 
        type=Path,
        default=None,
        help="A json file containing the training data."
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=None,
        help="A json file containing the testing data."
    )
    parser.add_argument(
        "--plot_file",
        type=Path,
        default=None,
        help="A jsonl file containing data used solely for plotting the learning curve."
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=10000,
        help="How much training data did you use?"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help=(
            "The maximum total sequence length after tokenization."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts",
            "polynomial", "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path,
        default=None,
        help="Where to store the final model."
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=11207330,
        help="A seed for reproducible training."
        )
    parser.add_argument(
        "--prediction_path",
        type=Path,
        default=Path("./predictions.txt"),
        help="Path to the output prediction file."
    )
    parser.add_argument(
        "--r",
        type=int,
        default=8,
        help="Lora attention dimension (the rank)."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="The alpha parameter for Lora scaling."
    )
    parser.add_argument(
        "--target_modules",
        nargs='*',
        default=None,
        help="The names of the modules to apply the adapter to."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot learning curves."
    )
    parser.add_argument(
        "--record_interval",
        type=int,
        default=250,
        help="Number of training steps between each recording of perplexity and loss."
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        args_dict = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).copy().items()
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "argument.json").write_text(json.dumps(args_dict, indent=4))

    return args


def preprocess_function(
    tokenizer: AutoTokenizer,
    data: List[Dict[str, str]],
    max_length: int = 2048,
    ) -> Dict[str, List[torch.Tensor]]:
    """preprocess_function"""
    data_size = len(data)
    instructions = [x["instruction"] for x in data]
    outputs = [x["output"] for x in data]

    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)

    for i in range(data_size):
        instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
        output_input_ids = tokenized_outputs["input_ids"][i] + [tokenizer.eos_token_id]
        tokenized_instructions["input_ids"][i] = instruction_input_ids + output_input_ids

        input_ids_length = len(tokenized_instructions["input_ids"][i])
        tokenized_instructions["attention_mask"][i] = [1] * input_ids_length

        tokenized_instructions["input_ids"][i] = torch.tensor(
            tokenized_instructions["input_ids"][i][:max_length]
        )
        tokenized_instructions["attention_mask"][i] = torch.tensor(
            tokenized_instructions["attention_mask"][i][:max_length]
        )

    return tokenized_instructions


def main():
    """main"""
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        quantization_config=get_bnb_config(),
    )
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path)
    else:
        peft_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    if args.plot_file and args.plot_file.exists():
        with args.plot_file.open("r", encoding="utf-8") as plot_file:
            plot_data = json.load(plot_file)
    else:
        plot_data = None

    evaluator = Evaluator(
        args.output_dir,
        tokenizer,
        data=plot_data,
        max_length=args.max_length,
        record_interval=args.record_interval,
    )

    if args.num_train_epochs > 0 and args.train_file and args.train_file.exists():
        with args.train_file.open("r", encoding="utf-8") as train_file:
            train_data = json.load(train_file)

        num_train_samples = len(train_data)
        if args.num_train_samples < num_train_samples:
            num_train_samples = args.num_train_samples
            train_data = random.sample(train_data, num_train_samples)

        train_dataset = preprocess_function(tokenizer, train_data, args.max_length)

        num_training_steps = (
            args.num_train_epochs * math.ceil(len(train_data) / args.gradient_accumulation_steps)
        )

        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
            num_training_steps=num_training_steps
        )
        model, optimizer, train_dataset, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataset, lr_scheduler
        )

        loss_fct = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(
            range(num_training_steps), disable=not accelerator.is_local_main_process
        )

        for _ in range(args.num_train_epochs):
            model.train()
            for i in range(num_train_samples):
                with accelerator.accumulate(model):
                    input_ids = train_dataset["input_ids"][i].unsqueeze(0)
                    attn_mask = train_dataset["attention_mask"][i].unsqueeze(0)

                    out_logits = model(input_ids, attention_mask=attn_mask).logits
                    shift_logits = out_logits[..., :-1, :].contiguous()
                    shift_label = input_ids[..., 1:].contiguous()

                    loss = loss_fct(shift_logits.transpose(1, 2), shift_label)

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    evaluator.add(loss.item(), model)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        for param in unwrapped_model.parameters():
            param.data = param.data.contiguous()

        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

        if args.plot:
            evaluator.plot_learning_curves()

    if args.test_file:
        with args.test_file.open("r", encoding="utf-8") as file:
            test_data = json.load(file)

        instructions = [get_prompt(x["instruction"]) for x in test_data]
        predictions = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(instructions, desc="Prediction", unit_scale=True, colour="red"):
                input_ids = tokenizer(
                    batch, return_tensors="pt", add_special_tokens=False
                )["input_ids"].to(accelerator.device)

                generated_tokens = model.generate(input_ids=input_ids, max_length=args.max_length)
                generated_tokens = generated_tokens.cpu().numpy()

                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                for decoded_input, decoded_pred in zip(decoded_inputs, decoded_preds):
                    cleaned_pred = decoded_pred[len(decoded_input):].strip().replace("�", "")
                    predictions.append(cleaned_pred)

        outputs = [
            {"id": item["id"], "output": prediction}
            for item, prediction in zip(test_data, predictions)
        ]

        with args.prediction_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

        print(f"\nThe prediction results have been saved to {args.prediction_path}")

if __name__ == "__main__":
    main()
