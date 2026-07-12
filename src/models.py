"""The five language models used for both surprisal and embeddings.

``last_layer`` is the index of the final hidden state in the
``output_hidden_states`` list (0 = input embeddings), i.e. the layer used in the
main analyses. ``sep`` is the tokenizer's sub-word boundary marker used to align
sub-word tokens back to whitespace words (``</w>`` for the original GPT,
``Ġ`` for the GPT-2 BPE family, which is loaded with ``add_prefix_space=True``).
"""

MODELS = {
    "gpt":         {"hf": "openai-gpt",  "dim": 768,  "last_layer": 12, "sep": "</w>", "cased": False},
    "gpt2_small":  {"hf": "gpt2",        "dim": 768,  "last_layer": 12, "sep": "Ġ",   "cased": True},
    "gpt2_medium": {"hf": "gpt2-medium", "dim": 1024, "last_layer": 24, "sep": "Ġ",   "cased": True},
    "gpt2_large":  {"hf": "gpt2-large",  "dim": 1280, "last_layer": 36, "sep": "Ġ",   "cased": True},
    "gpt2_xl":     {"hf": "gpt2-xl",     "dim": 1600, "last_layer": 48, "sep": "Ġ",   "cased": True},
}

# Order used across figures/tables and the paper's model-size axis.
MODEL_ORDER = ["gpt", "gpt2_small", "gpt2_medium", "gpt2_large", "gpt2_xl"]

# Pretty labels for figures (LaTeX math as in the original plots).
MODEL_LABELS = {
    "gpt": "GPT",
    "gpt2_small": r"GPT-2$_{124M}$",
    "gpt2_medium": r"GPT-2$_{355M}$",
    "gpt2_large": r"GPT-2$_{774M}$",
    "gpt2_xl": r"GPT-2$_{1.5B}$",
}
