from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract class for model generation."""

    def __init__(self, model_str, device="cpu"):
        self.model_str = model_str
        self.tokenizer = None
        self.device = device

    @abstractmethod
    def generate(self, prompt):
        """Method to generate entity state with model."""
