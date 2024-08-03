
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/multi_gpu.yaml \
    --num_processes=4 \
    --main_process_port 9505 \
    train_instruct.py recipes/deepseek/lora_instruct.yaml \
    # --torch_dtype=bfloat16 --bnb_4bit_quant_storage=bfloat16 \
    # --use_4bit_quantization=True \
    # --use_nested_quant=True \
    # --bnb_4bit_quant_type="nf4" \
    # --bnb_4bit_compute_dtype=bfloat16 \
    # --bnb_4bit_quant_storage_dtype=bfloat16
    