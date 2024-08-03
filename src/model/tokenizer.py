from transformers import AutoTokenizer, PreTrainedTokenizer
from src.alignment.configs import DataArguments, ModelArguments


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def get_tokenizer(model_args: ModelArguments, data_args: DataArguments) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
         trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    # if tokenizer.model_max_length > 100_000:
    #     tokenizer.model_max_length = 4096
    tokenizer.model_max_length = 4096

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def get_tokenizer_phi2(model_args: ModelArguments, data_args: DataArguments) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=False
    )
    tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
    tokenizer.pad_token = "<PAD>"
    tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))

    tokenizer.model_max_length = 2048
    

    return tokenizer

def get_tokenizer_qwen15(model_args: ModelArguments, data_args: DataArguments) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    # tokenizer.pad_token_id = tokenizer.eod_id

    # tokenizer.model_max_length = 8192

    return tokenizer


def get_tokenizer_code_llama(model_args: ModelArguments, data_args: DataArguments) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision
    )
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    return tokenizer