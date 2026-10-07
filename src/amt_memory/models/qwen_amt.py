"""One-token AMT prefix interface for a frozen Qwen reader.

This module tests H1 and H3 without retrieval or cross-attention: each supplied
gold postulate is pooled into exactly one continuous vector and prepended to the
query embeddings consumed by a frozen Qwen decoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor


class QwenPrefixAMT:
    """Map each postulate to one trainable prefix token for a frozen Qwen model."""

    def __init__(self, model: object, tokenizer: object) -> None:
        import torch
        from torch import nn

        self.model = model
        self.tokenizer = tokenizer
        self._torch = torch
        self.hidden_size = model.config.hidden_size
        self.embedding = model.get_input_embeddings()
        self.device = self.embedding.weight.device
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()

        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.projector.to(self.device)
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)

    def parameters(self):
        """Return only AMT parameters; the Qwen reader stays frozen."""
        return self.projector.parameters()

    def train(self) -> None:
        self.projector.train()
        self.model.eval()

    def eval(self) -> None:
        self.projector.eval()
        self.model.eval()

    def memory_tokens(self, postulates: Sequence[str]) -> Tensor:
        """Return one projected latent token per postulate, shape (1, M, D)."""
        if not postulates:
            raise ValueError("at least one gold postulate is required")
        encoded = self.tokenizer(
            list(postulates),
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        with self._torch.no_grad():
            embedded = self.embedding(encoded.input_ids)
            mask = encoded.attention_mask.unsqueeze(-1).to(embedded.dtype)
            pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projector(pooled).unsqueeze(0)

    def inputs(
        self, postulates: Sequence[str], query: str
    ) -> tuple[Tensor, Tensor]:
        """Create Qwen input embeddings with one AMT prefix per postulate."""
        query_inputs = self.tokenizer(
            query,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)
        query_embeddings = self.embedding(query_inputs.input_ids)
        memory_embeddings = self.memory_tokens(postulates).to(query_embeddings.dtype)
        embeddings = self._torch.cat((memory_embeddings, query_embeddings), dim=1)
        memory_mask = self._torch.ones(
            (1, memory_embeddings.shape[1]),
            dtype=query_inputs.attention_mask.dtype,
            device=self.device,
        )
        attention_mask = self._torch.cat(
            (memory_mask, query_inputs.attention_mask), dim=1
        )
        return embeddings, attention_mask

    def forward(
        self,
        postulates: Sequence[str],
        query: str,
        *,
        labels: Tensor | None = None,
    ) -> object:
        """Run frozen Qwen with AMT prefixes; labels cover query tokens only."""
        embeddings, attention_mask = self.inputs(postulates, query)
        if labels is not None:
            prefix_labels = self._torch.full(
                (labels.shape[0], embeddings.shape[1] - labels.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = self._torch.cat((prefix_labels, labels), dim=1)
        return self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels,
        )

    def loss(self, postulates: Sequence[str], prompt: str, answer: str) -> Tensor:
        """Compute answer-only next-token loss for one gold-memory example."""
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(self.device)
        answer_ids = self.tokenizer(
            answer, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        full_ids = self._torch.cat((prompt_ids, answer_ids), dim=1)
        labels = self._torch.cat(
            (
                self._torch.full_like(prompt_ids, -100),
                answer_ids,
            ),
            dim=1,
        )
        query_embeddings = self.embedding(full_ids)
        memory_embeddings = self.memory_tokens(postulates).to(query_embeddings.dtype)
        embeddings = self._torch.cat((memory_embeddings, query_embeddings), dim=1)
        attention_mask = self._torch.ones(
            embeddings.shape[:2], dtype=self._torch.long, device=self.device
        )
        prefix_labels = self._torch.full(
            (1, memory_embeddings.shape[1]),
            -100,
            dtype=labels.dtype,
            device=self.device,
        )
        return self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=self._torch.cat((prefix_labels, labels), dim=1),
        ).loss

    def generate(
        self,
        postulates: Sequence[str],
        query: str,
        *,
        max_new_tokens: int = 32,
    ) -> str:
        """Greedily answer from continuous AMT prefix tokens."""
        self.eval()
        embeddings, attention_mask = self.inputs(postulates, query)
        with self._torch.inference_mode():
            generated = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()