"""evaluate"""
import json
from pathlib import Path

import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class Evaluator:
    """Evaluator"""
    def __init__(
        self,
        output_dir: Path,
        tokenizer: AutoTokenizer,
        data: list = None,
        max_length: int = 2048,
        record_interval: int = 250,
        ) -> None:
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.history = []
        self.tokenized_instructions = None
        self.output_masks = None
        self.iteration_counter = 0
        self.train_loss_accumulator = 0
        self.record_interval = record_interval
        if data is not None:
            self.process_data(data, max_length)

    def process_data(
        self,
        data: list,
        max_length: int,
        ) -> None:
        """process_data"""
        data_size = len(data)
        instructions = [x["instruction"] for x in data]
        outputs = [x["output"] for x in data]

        tokenized_instructions = self.tokenizer(instructions, add_special_tokens=False)
        tokenized_outputs = self.tokenizer(outputs, add_special_tokens=False)
        output_masks = []

        for i in range(data_size):
            instruction_input_ids = tokenized_instructions["input_ids"][i]
            instruction_input_ids = [self.tokenizer.bos_token_id] + instruction_input_ids

            output_input_ids = tokenized_outputs["input_ids"][i] + [self.tokenizer.eos_token_id]
            tokenized_instructions["input_ids"][i] = instruction_input_ids + output_input_ids

            input_ids_length = len(tokenized_instructions["input_ids"][i])
            tokenized_instructions["attention_mask"][i] = [1] * input_ids_length
            output_mask = [0] * len(instruction_input_ids) + [1] * len(output_input_ids)

            tokenized_instructions["input_ids"][i] = torch.tensor(
                tokenized_instructions["input_ids"][i][:max_length]
            )
            tokenized_instructions["attention_mask"][i] = torch.tensor(
                tokenized_instructions["attention_mask"][i][:max_length]
            )
            output_masks.append(torch.tensor(output_mask[:max_length]))

        self.tokenized_instructions = tokenized_instructions
        self.output_masks = output_masks

    def calculate_ppl(self, model: AutoModelForCausalLM) -> dict:
        """calculate_ppl"""
        losses = []
        ppls = []

        mean_loss_fct = torch.nn.CrossEntropyLoss()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        for i in tqdm(range(len(self.output_masks))):
            input_ids = self.tokenized_instructions["input_ids"][i].unsqueeze(0)
            attn_mask = self.tokenized_instructions["attention_mask"][i].unsqueeze(0)
            output_mask = self.output_masks[i].unsqueeze(0)
            label = input_ids

            with torch.no_grad():
                out_logits = model(input_ids, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_label = label[..., 1:].contiguous()
            shift_output_mask = output_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_label)
                * shift_output_mask).sum(1) / shift_output_mask.sum(1)
            )
            ppls += perplexity_batch.tolist()
            losses += [mean_loss_fct(shift_logits.transpose(1, 2), shift_label)]

        return {
            "perplexities": ppls,
            "mean_perplexity": np.mean(ppls),
            "mean_loss": float(np.mean(losses))
        }

    def add(
        self,
        train_loss: float,
        model: AutoModelForCausalLM,
        ) -> None:
        """add"""
        self.iteration_counter += 1
        self.train_loss_accumulator += train_loss

        if self.iteration_counter % self.record_interval == 0:
            if self.tokenized_instructions is not None and self.output_masks is not None:
                ppl_data = self.calculate_ppl(model)
                mean_ppl = ppl_data["mean_perplexity"] if ppl_data else None
                mean_loss = ppl_data["mean_loss"] if ppl_data else None
            else:
                mean_ppl = None
                mean_loss = None

            average_train_loss = self.train_loss_accumulator / self.record_interval
            result = {
                "train_loss": average_train_loss,
                "mean_perplexity": mean_ppl,
                "mean_loss": mean_loss,
            }
            self.train_loss_accumulator = 0

            print(
                "\n"
                f"Train Loss: {average_train_loss:7.4f}, "
                f"Mean PPL: {mean_ppl:7.4f}" if mean_ppl else "N/A",
                f"Mean Loss: {mean_loss:7.4f}" if mean_loss else "N/A"
            )

            self.history.append(result)

    def plot_learning_curves(self) -> None:
        """plot_learning_curves"""
        import matplotlib.pyplot as plt

        instructions_count = [(i + 1) * self.record_interval for i in range(len(self.history))]
        train_losses = [entry["train_loss"] for entry in self.history]
        mean_perplexities = [
            entry["mean_perplexity"] for entry in self.history
            if entry["mean_perplexity"] is not None
        ]
        mean_losses = [
            entry["mean_loss"] for entry in self.history
            if entry["mean_loss"] is not None
        ]

        _, axes = plt.subplots(2, 1, figsize=(10, 12))

        if mean_perplexities:
            axes[0].plot(
                instructions_count, mean_perplexities,
                label="Mean Perplexity", marker='o', color='green'
            )
            axes[0].set_title("Perplexity History")
            axes[0].set_xlabel("Instructions")
            axes[0].set_ylabel("Score")
            axes[0].legend(loc='lower right')
            axes[0].grid(True)

        axes[1].plot(
            instructions_count, train_losses,
            label="Training Loss", marker='o', color='blue'
            )
        if mean_losses:
            axes[1].plot(
                instructions_count, mean_losses,
                label="Mean Loss", marker='x', color='red'
            )
        axes[1].set_title("Loss History")
        axes[1].set_xlabel("Instructions")
        axes[1].set_ylabel("Loss")
        axes[1].legend(loc='upper right')
        axes[1].grid(True)


        plt.tight_layout()

        if self.output_dir is not None:
            plt.savefig(self.output_dir / "learning_curves.png")
            plt.close()

        if self.output_dir is not None:
            history_file = self.output_dir / "history.json"
            with open(history_file, "w", encoding="utf-8") as file:
                json.dump(self.history, file, indent=4)
