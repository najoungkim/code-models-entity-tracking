from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract class for model generation."""

    def __init__(self, model_str, quantization_config=None, hf_key=None, device="cpu"):
        self.model_str = model_str
        self.model = None
        self.tokenizer = None
        self.device = device
        self.quantization_config = quantization_config
        self.hf_key = hf_key
        self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        """Method to initialize model (and tokenizer)."""

    @abstractmethod
    def generate(self, prompt):
        """Method to generate entity state with model."""
