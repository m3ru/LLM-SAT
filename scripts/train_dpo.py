"""
DPO Training Script for SAT Solver Code Generation
Usage: python train_dpo.py --config config.yaml
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_from_disk
import argparse

def setup_model_and_tokenizer(model_name, use_qlora=True):
    """Load model with QLoRA configuration"""
    
    # QLoRA configuration
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for DPO
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare for k-bit training
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA parameters"""
    return LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Adjust based on model architecture
    )

def main(args):
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_qlora=args.use_qlora
    )
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = load_from_disk(args.train_data)
    
    if args.eval_data:
        eval_dataset = load_from_disk(args.eval_data)
    else:
        # Split training data if no eval provided
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")
    
    # LoRA configuration
    peft_config = setup_lora_config()
    
    # DPO Training configuration
    training_args = DPOConfig(
        output_dir=args.output_dir,
        
        # Training hyperparameters
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        
        # DPO specific
        beta=args.beta,  # DPO temperature
        
        # Optimization
        optim="paged_adamw_32bit",  # Memory efficient optimizer
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Logging and saving
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        
        # Evaluation
        evaluation_strategy="steps",
        
        # Mixed precision
        bf16=True,
        
        # Other
        remove_unused_columns=False,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.run_name,
    )
    
    # Initialize DPO Trainer
    print("Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="deepseek-ai/deepseek-coder-7b-base-v1.5",
                        help="Base model to fine-tune")
    parser.add_argument("--use_qlora", action="store_true", default=True,
                        help="Use QLoRA for memory efficiency")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training dataset")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to evaluation dataset")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for model and checkpoints")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (overrides epochs)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    
    # Sequence length arguments
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default="sat-solver-dpo",
                        help="Run name for logging")
    
    args = parser.parse_args()
    main(args)
