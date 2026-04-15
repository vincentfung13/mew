import os
import torch
from omegaconf import OmegaConf

from mew.nn.utils import build_model
from mew.tokenization.bpe import BPETokenizer


class BaseConditionalGenerator:
    def __init__(self, model_path: str, device: str = "cuda"):
        # Fetch cfg
        cfg_path = os.path.join(os.path.dirname(model_path), "training_cfg.yaml")
        training_cfg = OmegaConf.load(cfg_path)

        # Init model
        self.device = device
        self.model = build_model(
            cfg=training_cfg,
        ).to(device)

        # Load state dict
        state_dict = torch.load(model_path, weights_only=False, map_location=device)[
            "model"
        ]
        self.model.load_state_dict(state_dict, strict=True)

        # Set to eval mode
        self.model.eval()

        # Load tokenizer
        tokenizer_dir = os.path.join(os.path.dirname(model_path), "tokenizer")
        self.tokenizer = BPETokenizer.from_dir(tokenizer_dir)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: int = 100,
        stop_token: str = "<|endoftext|>",
    ) -> str:
        # 1. Convert prompt to tokens
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)
        stop_token_ind = self.tokenizer.get_token_ind[stop_token.encode("utf-8")]

        # 2. Iteratively generate tokens
        for _ in range(max_new_tokens):
            # 2.1. Convert token list to tensor
            input_tensor = (
                torch.tensor(prompt_tokens, dtype=torch.long)
                .unsqueeze(0)
                .to(self.device)
            )

            # 2.2. Run through the model to obtain logits
            logits = self.model(input_tensor)  # (B, vocab_size)

            # 2.3. Compute softmax with temperature
            probs = torch.softmax(
                logits[:, -1] / temperature, dim=-1
            )  # (B, vocab_size)

            # 2.4. Sample from the distribution
            next_token_ind = torch.multinomial(probs, num_samples=1).item()
            # Check for stop token
            if next_token_ind == stop_token_ind:
                break
            prompt_tokens.append(next_token_ind)

        # Convert token list back to string
        output_str = self.tokenizer.decode(prompt_tokens[prompt_length:])

        return prompt + output_str
