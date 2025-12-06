"""
Quick test to verify setup before full training
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def test_setup():
    print("Testing setup...")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test model loading
    print("\nTesting model loading with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model_name = "deepseek-ai/deepseek-coder-7b-base-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    print("Model loaded successfully!")
    print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    # Test inference
    prompt = "def fibonacci(n):"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    print("\nTest generation:")
    print(tokenizer.decode(outputs[0]))
    
    print("\n✓ Setup verification complete!")

if __name__ == "__main__":
    test_setup()
