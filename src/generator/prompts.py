from typing import List, Tuple, Dict, Any


class PromptTemplate:
    """Template for generating prompts with in-context learning examples."""
    
    def __init__(self, task_description: str = "text classification"):
        """
        Initialize the prompt template.
        
        Args:
            task_description: Description of the classification task
        """
        self.task_description = task_description
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the task."""
        return f"""You are an expert at {self.task_description}. Your task is to generate high-quality text samples along with their correct labels.

Guidelines:
1. Always generate only one sample.
2. Generate realistic and diverse text samples
3. Ensure the text matches the given label
4. Vary the style, length, and content of your samples
5. Make the samples representative of real-world data
6. Always provide both the text and label in the specified format

Format your response as:
Text: [your generated text]
Label: [the corresponding label]"""
    
    def create_prompt(
        self, 
        icl_examples: List[Tuple[str, int, str]], 
        target_class: int = None
    ) -> str:
        """
        Create a prompt with in-context learning examples.
        
        Args:
            icl_examples: List of (text, label_id, label_name) examples
            target_class: Target class for generation (None for any class)
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = [self.system_prompt]
        prompt_parts.append("\nHere are some examples:")
        
        # Add ICL examples
        for text, label_id, label_name in icl_examples:
            prompt_parts.append(f"\nExample:")
            prompt_parts.append(f"Text: {text}")
            prompt_parts.append(f"Label: {label_name}")
        
        # Add generation instruction
        if target_class is not None:
            target_label = icl_examples[0][2] if icl_examples else "positive"  # fallback
            # Find the target label name from examples
            for _, label_id, label_name in icl_examples:
                if label_id == target_class:
                    target_label = label_name
                    break
            prompt_parts.append(f"\nNow generate a new text sample with label '{target_label}':")
        else:
            prompt_parts.append(f"\nNow generate a new text sample with any appropriate label:")
        
        return "\n".join(prompt_parts)
    
    def create_batch_prompts(
        self, 
        icl_examples: List[Tuple[str, int, str]], 
        num_samples: int,
        target_classes: List[int] = None
    ) -> List[str]:
        """
        Create multiple prompts for batch generation.
        
        Args:
            icl_examples: List of (text, label_id, label_name) examples
            num_samples: Number of samples to generate
            target_classes: List of target classes (None for random)
        
        Returns:
            List of formatted prompt strings
        """
        prompts = []
        
        for i in range(num_samples):
            if target_classes is not None and i < len(target_classes):
                target_class = target_classes[i]
            else:
                target_class = None
            
            prompt = self.create_prompt(icl_examples, target_class)
            print(prompt)
            prompts.append(prompt)
        
        return prompts
    
    def parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the model's response to extract text and label.
        
        Args:
            response: Raw response from the model
        
        Returns:
            Tuple of (text, label) or (None, None) if parsing fails
        """
        lines = response.strip().split('\n')
        text = None
        label = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Text:'):
                text = line[5:].strip()
            elif line.startswith('Label:'):
                label = line[6:].strip()
        
        return text, label
    
    def validate_response(self, text: str, label: str, valid_labels: List[str]) -> bool:
        """
        Validate that the parsed response is valid.
        
        Args:
            text: Extracted text
            label: Extracted label
            valid_labels: List of valid label names
        
        Returns:
            True if valid, False otherwise
        """
        if not text or not label:
            return False
        
        if len(text.strip()) < 10:  # Minimum text length
            return False
        
        if label not in valid_labels:
            return False
        
        return True


class ClassificationPromptTemplate(PromptTemplate):
    """Specialized prompt template for text classification tasks."""
    
    def __init__(self, task_description: str = "text classification", label_descriptions: Dict[str, str] = None):
        """
        Initialize the classification prompt template.
        
        Args:
            task_description: Description of the classification task
            label_descriptions: Dictionary mapping labels to descriptions
        """
        self.label_descriptions = label_descriptions or {}
        super().__init__(task_description)
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for classification tasks."""
        base_prompt = super()._create_system_prompt()
        
        if self.label_descriptions:
            base_prompt += "\n\nLabel descriptions:"
            for label, description in self.label_descriptions.items():
                base_prompt += f"\n- {label}: {description}"
        
        return base_prompt
    
    def create_prompt_with_label_info(
        self, 
        icl_examples: List[Tuple[str, int, str]], 
        target_class: int = None,
        include_label_info: bool = True
    ) -> str:
        """
        Create a prompt with additional label information.
        
        Args:
            icl_examples: List of (text, label_id, label_name) examples
            target_class: Target class for generation
            include_label_info: Whether to include label descriptions
        
        Returns:
            Formatted prompt string
        """
        if include_label_info and self.label_descriptions:
            # Add label information to the prompt
            prompt = self.create_prompt(icl_examples, target_class)
            
            # Insert label descriptions after examples
            lines = prompt.split('\n')
            insert_idx = -1
            for i, line in enumerate(lines):
                if line.startswith('Now generate'):
                    insert_idx = i
                    break
            
            if insert_idx > 0:
                label_info = "\n\nAvailable labels and their meanings:"
                for label, description in self.label_descriptions.items():
                    label_info += f"\n- {label}: {description}"
                
                lines.insert(insert_idx, label_info)
                return '\n'.join(lines)
        
        return self.create_prompt(icl_examples, target_class)

