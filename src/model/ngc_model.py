"""Implementation of conditional generation with NGC models."""


import requests
MODEL_STR_TO_API_KEY = {
    "llama2_70b": "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",
    "mixtral_8x7b_instruct": "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"
}


MODEL_STR_TO_URL = {
    "llama2_70b": 'https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0e349b44-440a-44e1-93e9-abe8dcb27158',
    "mixtral_8x7b_instruct": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/8f4118ba-60a8-4e6b-8574-e38a4067a4a3"
}


class NGCModel:
    """Wrapper class for Autoregressive models in HF."""
    fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"

    def __init__(self, model_str):
        self.model_str = model_str
        try:
            self.api_key = MODEL_STR_TO_API_KEY[model_str]
            self.invoke_url = MODEL_STR_TO_URL[model_str]
        except KeyError:
            print(f"{model_str} not in {MODEL_STR_TO_API_KEY.keys()}")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def generate(self, messages):
        """Generate response for chat-optimized model"""
        payload = {
            "messages": messages,
            "temperature": 0.01,
            "top_p": 0.95,
            "max_tokens": 1024,
            "seed": 42,
            "stream": False
        }

        session = requests.Session()
        response = session.post(
            self.invoke_url, headers=self.headers, json=payload)

        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            fetch_url = self.fetch_url_format + request_id
            response = session.get(fetch_url, headers=self.headers)

        response.raise_for_status()
        response_body = response.json()
        prediction = response_body['choices'][0]['message']['content']
        # print(prediction)

        return prediction
