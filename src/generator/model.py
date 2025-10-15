import torch
import torch.nn as nn
from transformers import (
    BitsAndBytesConfig,
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Tuple, Dict, Any, Optional
import logging
from tqdm import tqdm
from .prompts import PromptTemplate, ClassificationPromptTemplate


class LlamaGenerator:
    """Llama-3.1-8B generator with LoRA support for synthetic data generation."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        cache_dir: str = "./cache",
        quantization: str = "4bit",
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = "auto",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the Llama generator.
        
        Args:
            model_name: Name of the Llama model
            cache_dir: Directory to cache the model
            quantization: Quantization type ("4bit", "8bit", or "none")
            use_lora: Whether to use LoRA adapters
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            device: Device to use ("auto", "cuda", "cpu")
            dtype: Torch data type
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.quantization = quantization
        self.use_lora = use_lora
        self.device = self._get_device(device)
        self.dtype = dtype
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.prompt_template = None
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "down_proj"]
        ) if use_lora else None
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512,
            pad_token_id=None,  # Will be set after tokenizer loading
            eos_token_id=None,  # Will be set after tokenizer loading
        )
        
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load the model and tokenizer."""
        logging.info(f"Loading model: {self.model_name}")
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Update generation config with tokenizer info
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load model
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "dtype": self.dtype,
            "device_map": "auto" if self.device.type == "cuda" else None,
            "trust_remote_code": True
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Apply LoRA if enabled
        if self.use_lora and self.lora_config:
            self.model = get_peft_model(self.model, self.lora_config)
            logging.info("Applied LoRA adapters to the model")
        
        # Move to device if not using device_map
        if self.device.type == "cpu" or not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        logging.info(f"Model loaded successfully on {self.device}")
    
    def set_prompt_template(self, prompt_template: PromptTemplate):
        """Set the prompt template for generation."""
        self.prompt_template = prompt_template
    
    def generate_samples(
        self,
        icl_examples: List[Tuple[str, int, str]],
        num_samples: int,
        target_classes: List[int] = None,
        temperature: float = 0.7,
        max_length: int = 512,
        valid_labels: List[str] = None
    ) -> List[Tuple[str, int, str]]:
        """
        Generate synthetic samples using in-context learning.
        
        Args:
            icl_examples: List of (text, label_id, label_name) examples for ICL
            num_samples: Number of samples to generate
            target_classes: List of target classes (None for random)
            temperature: Sampling temperature
            max_length: Maximum generation length
            valid_labels: List of valid label names
        
        Returns:
            List of (text, label_id, label_name) tuples
        """
        if self.prompt_template is None:
            raise ValueError("Prompt template not set. Call set_prompt_template() first.")
        
        if valid_labels is None:
            valid_labels = list(set(label_name for _, _, label_name in icl_examples))
        
        # Create prompts
        prompts = self.prompt_template.create_batch_prompts(
            icl_examples, num_samples, target_classes
        )
        
        generated_samples = []

        # print(prompts)
        
        # Generate samples in batches to manage memory
        batch_size = 4  # Adjust based on GPU memory
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating samples"):
            batch_prompts = prompts[i:i + batch_size]
            batch_samples = self._generate_batch(
                batch_prompts, temperature, max_length, valid_labels, icl_examples
            )
            generated_samples.extend(batch_samples)

            print(generated_samples[-1])
        
        return generated_samples
    
    def _generate_batch(
        self,
        prompts: List[str],
        temperature: float,
        max_length: int,
        valid_labels: List[str],
        icl_examples: List[Tuple[str, int, str]]
    ) -> List[Tuple[str, int, str]]:
        """Generate a batch of samples."""
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Update generation config
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=256,  # Limit new tokens for efficiency
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and parse responses
        samples = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            generated_tokens = output[inputs['input_ids'][i].shape[0]:]
            response = self.tokenizer.decode(output, skip_special_tokens=True)

            print(response)
            
            # Parse response
            text, label = self.prompt_template.parse_response(response)
            
            # Validate and convert
            if self.prompt_template.validate_response(text, label, valid_labels):
                # Find label ID
                label_id = None
                for _, label_id_candidate, label_name in icl_examples:
                    if label_name == label:
                        label_id = label_id_candidate
                        break
                
                if label_id is not None:
                    samples.append((text, label_id, label))
        
        return samples
    
    def save_model(self, save_path: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logging.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a saved model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            dtype=self.dtype,
            device_map="auto" if self.device.type == "cuda" else None
        )
        logging.info(f"Model loaded from {load_path}")
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model configuration parameters."""
        return {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "use_lora": self.use_lora,
            "lora_config": self.lora_config,
            "device": str(self.device),
            "dtype": str(self.dtype)
        }
    
    def enable_training_mode(self):
        """Enable training mode for the model."""
        self.model.train()
        if self.use_lora:
            # Enable gradient computation for LoRA parameters
            for param in self.model.parameters():
                if param.requires_grad:
                    param.requires_grad = True
    
    def enable_eval_mode(self):
        """Enable evaluation mode for the model."""
        self.model.eval()
    
    def get_trainable_parameters(self) -> int:
        """Get the number of trainable parameters."""
        if self.use_lora:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())
