# Classical-Chinese-Translation

## 📚 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Inference](#inference)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

## 🌟 Overview
The Classical-Chinese-Translation project aims to fine-tune transformer models for bidirectional translation between Classical Chinese and modern languages. Utilizing advanced techniques like LoRA (Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning), the project optimizes translation performance with minimal computational resources. This work enables seamless translation between ancient texts and modern languages, bridging cultural and historical knowledge gaps..

## 💻 Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Shengwei-Peng/Classical-Chinese-Translation.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Classical-Chinese-Translation
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## ⚙️ Usage

To fine-tune the transformer model for Classical Chinese translation, follow these steps:

### 1. Preparing the Dataset
Ensure that your dataset is in JSON format, with each data entry containing an `instruction` and the expected `output`. Here's an example of the structure:
```json
{
    "id": "db63fb72-e211-4596-94a4-69617706f7ef",
    "instruction": "翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案：",
    "output": "雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
}
```

### 2. Fine-Tuning the Model
To fine-tune the model, use the following command:
```sh
python main.py \
    --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
    --train_file ./data/train.json \
    --output_dir ./gemma_2_2b \
    --seed 11207330 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --num_train_samples 10000 \
    --gradient_accumulation_steps 4 \
    --r 8 \
    --lora_alpha 8 \
    --target_modules v_proj q_proj \
    --record_interval 250
```
If you want to plot the learning curves during training (such as loss or perplexity), you can add the `--plot_file` and `--plot` arguments to your command:
```sh
python main.py \
    --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
    --train_file ./data/train.json \
    --plot_file ./data/public_test.json \
    --output_dir ./gemma_2_2b \
    --seed 11207330 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --num_train_samples 10000 \
    --gradient_accumulation_steps 4 \
    --r 8 \
    --lora_alpha 8 \
    --target_modules v_proj q_proj \
    --record_interval 250 \
    --plot
```
#### ⚠️ Special Note: 
To enable plotting, you need to install the `matplotlib` library. You can do so by running:
```sh
pip install matplotlib
```

## 🔮 Inference
To generate predictions using the fine-tuned model, use the following command:
```sh
python main.py \
    --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
    --peft_path ./gemma_2_2b \
    --test_file ./data/private_test.json \
    --prediction_path ./prediction.json
```
The predictions will be free of any special tokens (e.g., `<s>`, `</s>`) and prompts, and the output format will look like this:
```json
{
    "id": "0094a447412998f6",
    "output": "高祖初年，任內祕書侍禦中散。"
}
```

## 🙏 Acknowledgements

This project is based on the example code provided by Hugging Face in their [Transformers repository](https://github.com/huggingface/transformers/tree/main/examples/pytorch). We have made modifications to adapt the code for our specific use case.

Special thanks to the [NTU Miulab](http://adl.miulab.tw) professors and teaching assistants for providing the dataset and offering invaluable support throughout the project.

## ⚖️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

## ✉️ Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw