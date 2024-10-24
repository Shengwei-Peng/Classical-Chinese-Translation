"""utils"""
from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str, mode: str = None) -> str:
    '''Format the instruction as a prompt for LLM.'''
    examples = [
        {
            "instruction": "翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。\n答案：",
            "output": "雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
        },
        {
            "instruction": "太祖親自擬文，然後嵌在上麵，是： 上天保佑人民，朕纔能夠統率安撫。\n這句話在中國古代怎麼說：",
            "output": "太祖親為文鈒之曰： 上天祐民，朕乃率撫。"
        },
        {
            "instruction": "文言文翻譯：\n是年，遷北院樞密副使，復陳陰害太子計，乙辛從之。",
            "output": "答案：同年，十三遷任北院樞密副使，又陳說暗中謀害太子的計策，乙辛從之。"
        },
        {
            "instruction": "今聽其所執，以壽光公就第，位同保傅，在三司之右。\n翻譯成現代文：",
            "output": "如今聽從他的意願，以壽光公身份迴宅第，地位如同太保太傅，在三司之上。"
        },
        {
            "instruction": "將下麵句子翻譯成現代文：\n繼踵鬍母，仲舒雲盛，因修《榖梁》，韆鞦最篤。",
            "output": "繼承跟踵鬍母生，董仲舒達到頂盛，因循《榖梁》，夏侯韆鞦最忠誠。"
        },
    ]

    if mode == "zero-shot":
        prompt = (
            "你是一位技術精湛的翻譯家，精通古典漢語（文言文、古文）。"
            "你的任務是將提供的現代中文文字翻譯成古典中文，反之亦然，具體取決於說明。"
            "保留翻譯中的原始含義、語氣和細微差別至關重要。"
            "確保翻譯文本保持正確的語法、拼字和標點符號，並遵守文言文的文體慣例。\n"
            f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        )

    elif mode == "few-shot":
        prompt = "".join(
            f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n"
            f"<start_of_turn>model\n{example['output']}<end_of_turn>\n"
            for example in examples
        )
        prompt += f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

    else:
        prompt = instruction

    return prompt


def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
