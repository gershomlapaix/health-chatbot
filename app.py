import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "./artifacts/healthbot_tinyllama_lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, LORA_PATH)

# Merge LoRA weights with base model (important for inference)
model = model.merge_and_unload()

model.eval()

def generate_response(message, history):
    prompt = f"<|user|>\n{message}\n<|assistant|>"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt from output
    response = response.split("<|assistant|>")[-1].strip()
    
    return response

demo = gr.ChatInterface(
    fn=generate_response,
    title="HealthBot - AI Health Assistant",
    description="""
    This is a demo health chatbot.
    """
)

if __name__ == "__main__":
    demo.launch()