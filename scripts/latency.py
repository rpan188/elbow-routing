import torch
import torch.nn.functional as F
import time
import numpy as np
import pickle
import argparse

from transformers import OlmoeForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset


DEVICE = "cuda"

# Load different ckpts 
model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")


latency_stats = {
    "elbow_forward_times": [],
    "original_forward_times": [],
}


def _cuda_sync_if_needed(x: torch.Tensor):
    if x.is_cuda:
        torch.cuda.synchronize()

def _top_k_elbow(routing_weights: torch.Tensor):
    device = routing_weights.device
    dtype = routing_weights.dtype
    N, E = routing_weights.shape

    sorted_vals, sorted_idx = torch.sort(routing_weights, dim=-1, descending=True)

    x_norm = torch.linspace(0, 1, E, device=device, dtype=dtype)

    y_first = sorted_vals[:, 0:1]
    y_last = sorted_vals[:, -1:]
    y_norm = (y_first - sorted_vals) / (y_first - y_last + 1e-12)

    elbow_scores = y_norm - x_norm.unsqueeze(0)
    elbow_indices = torch.argmax(elbow_scores, dim=1)

    ks = torch.clamp(elbow_indices + 1, min=1, max=8)

    MAX_K = 8
    mask = torch.arange(MAX_K, device=device).unsqueeze(0) < ks.unsqueeze(1)

    top_k_weights = sorted_vals[:, :MAX_K] * mask
    top_k_index = sorted_idx[:, :MAX_K]

    return top_k_weights, top_k_index

def forward_with_elbow_instrumented(self, hidden_states: torch.Tensor):
    _cuda_sync_if_needed(hidden_states)
    start = time.perf_counter()

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

    routing_weights, selected_experts = _top_k_elbow(routing_weights)

    if getattr(self, "norm_topk_prob", False):
        denom = routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routing_weights = routing_weights / denom

    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    #this one_hot uses num_experts = self.num_experts if available, else len(self.experts)
    num_experts = getattr(self, "num_experts", len(self.experts))

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)

    for expert_idx in range(num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])
        if top_x.numel() == 0:
            continue

        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    _cuda_sync_if_needed(final_hidden_states)
    latency_stats["elbow_forward_times"].append(time.perf_counter() - start)

    return final_hidden_states, router_logits

def _get_moe_block():
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    return OlmoeSparseMoeBlock

def use_original_forward():
    OlmoeSparseMoeBlock = _get_moe_block()

    # Save true original exactly once
    if not hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock._forward_true_original = OlmoeSparseMoeBlock.forward

    # Avoid double-wrapping
    if getattr(OlmoeSparseMoeBlock.forward, "__name__", "") == "forward_original_timed":
        print("Already using ORIGINAL forward (timed)")
        return

    def forward_original_timed(self, hidden_states: torch.Tensor, *args, **kwargs):
        _cuda_sync_if_needed(hidden_states)
        start = time.perf_counter()

        out = OlmoeSparseMoeBlock._forward_true_original(self, hidden_states, *args, **kwargs)

        # out could be Tensor or (Tensor, router_logits); synchronize on the tensor
        if isinstance(out, tuple):
            _cuda_sync_if_needed(out[0])
        else:
            _cuda_sync_if_needed(out)

        latency_stats["original_forward_times"].append(time.perf_counter() - start)
        return out

    OlmoeSparseMoeBlock.forward = forward_original_timed
    print("Switched to ORIGINAL forward (timed)")

def use_dynamic_forward():
    OlmoeSparseMoeBlock = _get_moe_block()

    if not hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock._forward_true_original = OlmoeSparseMoeBlock.forward

    OlmoeSparseMoeBlock.forward = forward_with_elbow_instrumented
    print("Switched to ELBOW top-k forward (timed)")

def reset_forward():
    OlmoeSparseMoeBlock = _get_moe_block()
    if hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock.forward = OlmoeSparseMoeBlock._forward_true_original
        print("Reset to true original forward")
    else:
        print("No saved original forward found (call use_* once first)")

def reset_stats():
    latency_stats["elbow_forward_times"] = []
    latency_stats["original_forward_times"] = []
    print("Statistics reset")

def compare_latencies():
    lat = latency_stats

    print("\n" + "=" * 70)
    print("LATENCY COMPARISON: Elbow Top-K vs Original")
    print("=" * 70)

    if lat["elbow_forward_times"]:
        d = np.array(lat["elbow_forward_times"]) * 1000
        print("\nElbow Top-K:")
        print(f"  Mean: {d.mean():.3f} ms ± {d.std():.3f} ms")
        print(f"  Total: {d.sum():.1f} ms")
        print(f"  Passes: {len(d)}")

    if lat["original_forward_times"]:
        o = np.array(lat["original_forward_times"]) * 1000
        print("\nOriginal:")
        print(f"  Mean: {o.mean():.3f} ms ± {o.std():.3f} ms")
        print(f"  Total: {o.sum():.1f} ms")
        print(f"  Passes: {len(o)}")

    if lat["elbow_forward_times"] and lat["original_forward_times"]:
        d_mean = np.mean(lat["elbow_forward_times"]) * 1000
        o_mean = np.mean(lat["original_forward_times"]) * 1000
        speed = o_mean / d_mean
        overhead = (d_mean - o_mean) / o_mean * 100.0

        print("\n" + "=" * 70)
        if speed > 1:
            print(f"  Elbow-based routing is {speed:.2f}x FASTER")
        else:
            print(f"  Elbow-based routing is {1/speed:.2f}x SLOWER")
        print(f"  Overhead: {overhead:+.2f}%")
        print("=" * 70)

def load_dataset_by_name(benchmark_name):
    if benchmark_name == "mmlu":
        return load_dataset("cais/mmlu", "all", split='test')
    elif benchmark_name == "arc_easy":
        return load_dataset("ai2_arc", "ARC-Easy", split='test')
    elif benchmark_name == "arc_challenge":
        return load_dataset("ai2_arc", "ARC-Challenge", split='test')
    elif benchmark_name == "hellaswag":
        return load_dataset("hellaswag", split='validation')
    elif benchmark_name == "piqa":
        return load_dataset("piqa", split='validation')
    elif benchmark_name == "winogrande":
        return load_dataset("winogrande", "winogrande_xl", split='validation')
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

def format_mmlu_prompt(example):
    """Format MMLU example as a prompt"""
    question = example["question"]
    choices = example["choices"]
    
    # Format as multiple choice
    prompt = f"Question: {question}\n"
    prompt += "Choices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"  # A, B, C, D
    prompt += "Answer:"
    
    return prompt

def format_arc_prompt(example):
    """Format an AI2 ARC example as a multiple-choice prompt.

    Works with common HF `ai2_arc` schemas, including:
    - example["question"]["stem"] + example["question"]["choices"]
    - example["question"] + example["choices"]
    """
    q = example.get("question")
    if isinstance(q, dict):
        question = q.get("stem", "")
        raw_choices = q.get("choices")
    else:
        question = q
        raw_choices = example.get("choices")

    pairs = []
    if isinstance(raw_choices, dict) and "label" in raw_choices and "text" in raw_choices:
        labels = raw_choices.get("label") or []
        texts = raw_choices.get("text") or []
        pairs = list(zip(labels, texts))
    elif isinstance(raw_choices, list):
        for item in raw_choices:
            if isinstance(item, dict):
                pairs.append((item.get("label"), item.get("text")))
            else:
                pairs.append((None, str(item)))

    prompt = f"Question: {question}\n"
    prompt += "Choices:\n"
    for i, (label, text) in enumerate(pairs):
        choice_label = label if label else chr(65 + i)
        prompt += f"{choice_label}. {text}\n"
    prompt += "Answer:"
    return prompt

def format_hellaswag_prompt(example):
    """Format HellaSwag example as a prompt."""
    ctx = example.get("ctx") or example.get("context") or ""
    endings = example.get("endings") or example.get("choices") or []

    prompt = f"Context: {ctx}\n"
    prompt += "Choices:\n"
    for i, ending in enumerate(endings):
        prompt += f"{chr(65+i)}. {ending}\n"
    prompt += "Answer:"
    return prompt

def format_piqa_prompt(example):
    """Format PIQA example as a prompt."""
    goal = example.get("goal") or ""
    sol1 = example.get("sol1") or ""
    sol2 = example.get("sol2") or ""

    prompt = f"Goal: {goal}\n"
    prompt += "Choices:\n"
    prompt += f"A. {sol1}\n"
    prompt += f"B. {sol2}\n"
    prompt += "Answer:"
    return prompt

def format_winogrande_prompt(example):
    """Format WinoGrande example as a prompt."""
    sentence = example.get("sentence") or ""
    option1 = example.get("option1") or ""
    option2 = example.get("option2") or ""

    prompt = f"Sentence: {sentence}\n"
    prompt += "Choices:\n"
    prompt += f"A. {option1}\n"
    prompt += f"B. {option2}\n"
    prompt += "Answer:"
    return prompt

PROMPT_FORMATTERS = {
    "mmlu": format_mmlu_prompt,
    "arc_easy": format_arc_prompt,
    "arc_challenge": format_arc_prompt,
    "hellaswag": format_hellaswag_prompt,
    "piqa": format_piqa_prompt,
    "winogrande": format_winogrande_prompt,
}

def get_prompt_formatter(benchmark: str):
    return PROMPT_FORMATTERS[benchmark]

def run_batch(samples, format_prompt_fn=format_mmlu_prompt, batch_size=20):
    """Process samples in batches"""
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        prompts = [format_prompt_fn(ex) for ex in batch]
        
        # Tokenize batch
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        if i % 40 == 0:
            print(f"Processed {i}/{len(samples)} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OLMoE benchmark")
    parser.add_argument("--method", default="original",
                        choices=["original", "elbow"],
                        help="Selection method to use (default: original)")
    parser.add_argument("--benchmark", default="mmlu", 
                        choices=["arc_easy", "arc_challenge", "mmlu", "hellaswag", "piqa", "winogrande"],
                        help="Benchmark to run (default: mmlu)")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to run (default: full test/validation split)",
    )
    args = parser.parse_args()

    # Load benchmark
    test_samples = load_dataset_by_name(args.benchmark)

    # Get a subset for testing (e.g., 100 examples from validation set)
    if args.num_samples is not None:
        n = min(args.num_samples, len(test_samples))
        test_samples = test_samples.select(range(n))

    # Convert to list
    test_samples_list = [ex for ex in test_samples]

    # Get prompt formatter corresponding to benchmark
    format_prompt_fn = get_prompt_formatter(args.benchmark)
    # Test formatting
    print(format_prompt_fn(test_samples[0]))

    if args.method == "elbow":
        # Test with elbow
        print("Testing ELBOW with batching...")
        use_dynamic_forward()
        run_batch(test_samples_list, format_prompt_fn=format_prompt_fn)
        # save
        with open(f"{args.benchmark}_elbow.pkl", "wb") as f:
            pickle.dump(latency_stats["elbow_forward_times"], f)
    elif args.method == "original":
        # Test with original
        use_original_forward()
        print("\nTesting ORIGINAL with batching...")
        run_batch(test_samples_list, format_prompt_fn=format_prompt_fn)
        # save
        with open(f"{args.benchmark}_original.pkl", "wb") as f:
            pickle.dump(latency_stats["original_forward_times"], f)

    compare_latencies()
