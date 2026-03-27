import re
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


# Initialize with empty dict - will be populated per-layer
elbow_topk_stats = {}


def calculate_moe_flops(batch_size, seq_len, hidden_dim, intermediate_dim, num_experts, avg_k):
    """Calculate FLOPs for MoE block."""
    N = batch_size * seq_len
    
    # Router FLOPs
    router_linear = 2 * N * hidden_dim * num_experts
    router_softmax = 5 * N * num_experts
    router_topk = N * num_experts * np.log2(num_experts)
    
    # Expert FLOPs
    expert_up = 2 * N * avg_k * hidden_dim * intermediate_dim
    expert_down = 2 * N * avg_k * intermediate_dim * hidden_dim
    expert_activation = N * avg_k * intermediate_dim
    
    total = router_linear + router_softmax + router_topk + expert_up + expert_down + expert_activation
    
    return total

def calculate_elbow_topk_overhead(batch_size, seq_len, num_experts):
    """Additional FLOPs for elbow top-k detection."""
    N = batch_size * seq_len
    
    sort_flops = N * num_experts * np.log2(num_experts)
    normalize_flops = 4 * N * num_experts
    subtract_flops = N * num_experts
    argmax_flops = N * num_experts
    
    total = sort_flops + normalize_flops + subtract_flops + argmax_flops
    
    return total

def _top_k_elbow_fast(routing_weights: torch.Tensor):
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

    return top_k_weights, top_k_index, ks

def forward_with_elbow_topk_instrumented(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    is_prefill = sequence_length > 1

    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

    routing_weights, selected_experts, ks = _top_k_elbow_fast(routing_weights)

    if getattr(self, "norm_topk_prob", False):
        denom = routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routing_weights = routing_weights / denom

    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    num_experts = getattr(self, "num_experts", len(self.experts))

    # ========= NEW: expert usage via bincount (no one_hot) =========
    Kmax = selected_experts.size(1)  # e.g., 8
    active = (torch.arange(Kmax, device=selected_experts.device)[None, :] < ks[:, None]).reshape(-1)
    flat_selected = selected_experts.reshape(-1)[active]
    expert_usage_counts = torch.bincount(flat_selected, minlength=num_experts).to("cpu")
    # ===============================================================

    # Keep your existing expert computation (still needs routing_weights + selected_experts)
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

    # Calculate average k and FLOPs (rough)
    avg_k = ks.float().mean().item()
    intermediate_dim = hidden_dim * 4

    base_flops = calculate_moe_flops(
        batch_size, sequence_length, hidden_dim, intermediate_dim,
        num_experts, avg_k
    )

    overhead_flops = calculate_elbow_topk_overhead(
        batch_size, sequence_length, num_experts
    )

    total_flops = base_flops + overhead_flops

    # Save stats
    layer_id = int(getattr(self, "layer_idx", -1))

    if layer_id >= 0:
        if layer_id not in elbow_topk_stats:
            elbow_topk_stats[layer_id] = {
                "k_prefill": [],
                "k_decode": [],
                "flops_prefill": [],
                "flops_decode": [],
                "original_flops_prefill": [],
                "original_flops_decode": [],
                "expert_usage_prefill": [],
                "expert_usage_decode": [],
                "original_expert_usage_prefill": [],
                "original_expert_usage_decode": []
            }

        if is_prefill:
            elbow_topk_stats[layer_id]["k_prefill"].append(ks.cpu())
            elbow_topk_stats[layer_id]["flops_prefill"].append(total_flops)
            elbow_topk_stats[layer_id]["expert_usage_prefill"].append(expert_usage_counts)
        else:
            elbow_topk_stats[layer_id]["k_decode"].append(ks.cpu())
            elbow_topk_stats[layer_id]["flops_decode"].append(total_flops)
            elbow_topk_stats[layer_id]["expert_usage_decode"].append(expert_usage_counts)

    return final_hidden_states, router_logits

def _get_moe_block():
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    return OlmoeSparseMoeBlock

def use_original_forward():
    OlmoeSparseMoeBlock = _get_moe_block()

    if not hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock._forward_true_original = OlmoeSparseMoeBlock.forward

    def forward_original_track_flops(self, hidden_states: torch.Tensor, *args, **kwargs):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        is_prefill = sequence_length > 1

        intermediate_dim = hidden_dim * 4
        num_experts = getattr(self, "num_experts", len(self.experts))
        k = int(getattr(self, "top_k", getattr(self, "num_experts_per_tok", 8)))  # NEW: not hard-coded

        base_flops = calculate_moe_flops(
            batch_size, sequence_length, hidden_dim, intermediate_dim,
            num_experts, k
        )

        layer_id = int(getattr(self, "layer_idx", -1))

        if layer_id >= 0:
            if layer_id not in elbow_topk_stats:
                elbow_topk_stats[layer_id] = {
                    "k_prefill": [],
                    "k_decode": [],
                    "flops_prefill": [],
                    "flops_decode": [],
                    "original_flops_prefill": [],
                    "original_flops_decode": [],
                    "expert_usage_prefill": [],
                    "expert_usage_decode": [],
                    "original_expert_usage_prefill": [],
                    "original_expert_usage_decode": []
                }

            if is_prefill:
                elbow_topk_stats[layer_id]["original_flops_prefill"].append(base_flops)
            else:
                elbow_topk_stats[layer_id]["original_flops_decode"].append(base_flops)

        # Call original forward
        out = OlmoeSparseMoeBlock._forward_true_original(self, hidden_states, *args, **kwargs)

        # ========= NEW: original expert usage via bincount (no per-expert loop) =========
        if layer_id >= 0:
            hidden_states_flat = hidden_states.view(-1, hidden_dim)
            with torch.no_grad():
                router_logits = self.gate(hidden_states_flat)
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                _, top_k_index = torch.topk(routing_weights, k=k, dim=-1)  # [N, k]
                expert_usage_counts = torch.bincount(top_k_index.reshape(-1), minlength=num_experts).to("cpu")

                if is_prefill:
                    elbow_topk_stats[layer_id]["original_expert_usage_prefill"].append(expert_usage_counts)
                else:
                    elbow_topk_stats[layer_id]["original_expert_usage_decode"].append(expert_usage_counts)
        # ==============================================================================

        return out

    OlmoeSparseMoeBlock.forward = forward_original_track_flops
    print("✓ Switched to ORIGINAL forward (tracking FLOPs and expert usage with phase separation)")

def use_elbow_forward():
    OlmoeSparseMoeBlock = _get_moe_block()

    if not hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock._forward_true_original = OlmoeSparseMoeBlock.forward

    OlmoeSparseMoeBlock.forward = forward_with_elbow_topk_instrumented
    print("✓ Switched to ELBOW top-k forward (tracking k and FLOPs with phase separation)")

def reset_forward():
    OlmoeSparseMoeBlock = _get_moe_block()
    if hasattr(OlmoeSparseMoeBlock, "_forward_true_original"):
        OlmoeSparseMoeBlock.forward = OlmoeSparseMoeBlock._forward_true_original
        print("✓ Reset to true original forward")
    else:
        print("⚠ No saved original forward found (call use_* once first)")

def reset_stats():
    """Clear all statistics."""
    global elbow_topk_stats
    elbow_topk_stats = {}
    print("✓ Statistics reset")

def compare_k_and_flops():
    """Compare k values and FLOPs between elbow and original, separated by phase."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: Elbow Top-K vs Original")
    print("=" * 70)
    
    # Collect stats
    elbow_k_prefill = []
    elbow_k_decode = []
    elbow_flops_prefill = []
    elbow_flops_decode = []
    original_flops_prefill = []
    original_flops_decode = []
    
    # Gather from per-layer stats
    for key in elbow_topk_stats.keys():
        if isinstance(key, int):
            layer_data = elbow_topk_stats[key]
            if isinstance(layer_data, dict):
                # K values (only elbow has these)
                if "k_prefill" in layer_data:
                    for k_tensor in layer_data["k_prefill"]:
                        if isinstance(k_tensor, torch.Tensor):
                            elbow_k_prefill.extend(k_tensor.flatten().tolist())
                        else:
                            elbow_k_prefill.append(k_tensor)
                
                if "k_decode" in layer_data:
                    for k_tensor in layer_data["k_decode"]:
                        if isinstance(k_tensor, torch.Tensor):
                            elbow_k_decode.extend(k_tensor.flatten().tolist())
                        else:
                            elbow_k_decode.append(k_tensor)
                
                # Elbow FLOPs
                if "flops_prefill" in layer_data:
                    elbow_flops_prefill.extend(layer_data["flops_prefill"])
                if "flops_decode" in layer_data:
                    elbow_flops_decode.extend(layer_data["flops_decode"])
                
                # Original FLOPs
                if "original_flops_prefill" in layer_data:
                    original_flops_prefill.extend(layer_data["original_flops_prefill"])
                if "original_flops_decode" in layer_data:
                    original_flops_decode.extend(layer_data["original_flops_decode"])
    
    # ============ PREFILL PHASE ============
    print("\n" + "PREFILL PHASE (Processing Input Prompt)")
    print("-" * 70)
    
    if elbow_k_prefill:
        d_k = np.array(elbow_k_prefill)
        print(f"\nK VALUES - Elbow Top-K (Prefill):")
        print(f"  Mean: {d_k.mean():.3f} ± {d_k.std():.3f}")
        print(f"  Min: {d_k.min():.1f}, Max: {d_k.max():.1f}")
        print(f"  Samples: {len(d_k)}")
    
    print(f"\nK VALUES - Original (Prefill):")
    print(f"  Constant: k = 8.000")
    
    if elbow_flops_prefill:
        d_flops = np.array(elbow_flops_prefill)
        print(f"\nFLOPs - Elbow Top-K (Prefill):")
        print(f"  Mean per pass: {d_flops.mean()/1e9:.3f} GFLOPs")
        print(f"  Total: {d_flops.sum()/1e12:.3f} TFLOPs")
        print(f"  Passes: {len(d_flops)}")
    
    if original_flops_prefill:
        o_flops = np.array(original_flops_prefill)
        print(f"\nFLOPs - Original (Prefill):")
        print(f"  Mean per pass: {o_flops.mean()/1e9:.3f} GFLOPs")
        print(f"  Total: {o_flops.sum()/1e12:.3f} TFLOPs")
        print(f"  Passes: {len(o_flops)}")
    
    # Prefill comparison
    if elbow_k_prefill:
        d_k = np.array(elbow_k_prefill)
        k_ratio = d_k.mean() / 8.0
        k_savings = ((8.0 - d_k.mean()) / 8.0) * 100
        print(f"\nPREFILL K COMPARISON:")
        print(f"  Elbow uses {k_ratio:.2f}x experts on average (vs 8)")
        if k_savings > 0:
            print(f"  Experts saved: {k_savings:.1f}%")
        else:
            print(f"  Extra experts used: {-k_savings:.1f}%")
    
    if elbow_flops_prefill and original_flops_prefill:
        d_flops = np.array(elbow_flops_prefill)
        o_flops = np.array(original_flops_prefill)
        flop_ratio = d_flops.mean() / o_flops.mean()
        flop_savings = ((o_flops.mean() - d_flops.mean()) / o_flops.mean()) * 100
        
        print(f"\nPREFILL FLOPs COMPARISON:")
        if flop_ratio < 1:
            print(f"  Elbow uses {flop_ratio:.2f}x FLOPs (FEWER)")
            print(f"  FLOPs saved: {flop_savings:.1f}%")
        else:
            print(f"  Elbow uses {flop_ratio:.2f}x FLOPs (MORE)")
            print(f"  FLOPs overhead: {-flop_savings:.1f}%")
    
    # ============ DECODE PHASE ============
    print("\n" + "DECODE PHASE (Generating New Tokens)")
    print("-" * 70)
    
    if elbow_k_decode:
        d_k = np.array(elbow_k_decode)
        print(f"\nK VALUES - Elbow Top-K (Decode):")
        print(f"  Mean: {d_k.mean():.3f} ± {d_k.std():.3f}")
        print(f"  Min: {d_k.min():.1f}, Max: {d_k.max():.1f}")
        print(f"  Samples: {len(d_k)}")
    
    print(f"\nK VALUES - Original (Decode):")
    print(f"  Constant: k = 8.000")
    
    if elbow_flops_decode:
        d_flops = np.array(elbow_flops_decode)
        print(f"\nFLOPs - Elbow Top-K (Decode):")
        print(f"  Mean per pass: {d_flops.mean()/1e9:.3f} GFLOPs")
        print(f"  Total: {d_flops.sum()/1e12:.3f} TFLOPs")
        print(f"  Passes: {len(d_flops)}")
    
    if original_flops_decode:
        o_flops = np.array(original_flops_decode)
        print(f"\nFLOPs - Original (Decode):")
        print(f"  Mean per pass: {o_flops.mean()/1e9:.3f} GFLOPs")
        print(f"  Total: {o_flops.sum()/1e12:.3f} TFLOPs")
        print(f"  Passes: {len(o_flops)}")
    
    # Decode comparison
    if elbow_k_decode:
        d_k = np.array(elbow_k_decode)
        k_ratio = d_k.mean() / 8.0
        k_savings = ((8.0 - d_k.mean()) / 8.0) * 100
        print(f"\nDECODE K COMPARISON:")
        print(f"  Elbow uses {k_ratio:.2f}x experts on average (vs 8)")
        if k_savings > 0:
            print(f"  Experts saved: {k_savings:.1f}%")
        else:
            print(f"  Extra experts used: {-k_savings:.1f}%")
    
    if elbow_flops_decode and original_flops_decode:
        d_flops = np.array(elbow_flops_decode)
        o_flops = np.array(original_flops_decode)
        flop_ratio = d_flops.mean() / o_flops.mean()
        flop_savings = ((o_flops.mean() - d_flops.mean()) / o_flops.mean()) * 100
        
        print(f"\nDECODE FLOPs COMPARISON:")
        if flop_ratio < 1:
            print(f"  Elbow uses {flop_ratio:.2f}x FLOPs (FEWER)")
            print(f"  FLOPs saved: {flop_savings:.1f}%")
        else:
            print(f"  Elbow uses {flop_ratio:.2f}x FLOPs (MORE)")
            print(f"  FLOPs overhead: {-flop_savings:.1f}%")
    
    # ============ OVERALL SUMMARY ============
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    # Combined stats
    all_elbow_k = elbow_k_prefill + elbow_k_decode
    all_elbow_flops = elbow_flops_prefill + elbow_flops_decode
    all_original_flops = original_flops_prefill + original_flops_decode
    
    if all_elbow_k:
        d_k_all = np.array(all_elbow_k)
        print(f"\nOverall Elbow K: {d_k_all.mean():.3f} ± {d_k_all.std():.3f}")
    
    if all_elbow_flops and all_original_flops:
        d_flops_all = np.array(all_elbow_flops)
        o_flops_all = np.array(all_original_flops)
        
        print(f"\nTotal FLOPs:")
        print(f"  Elbow:    {d_flops_all.sum()/1e12:.3f} TFLOPs")
        print(f"  Original: {o_flops_all.sum()/1e12:.3f} TFLOPs")
        print(f"  Savings:  {((o_flops_all.sum() - d_flops_all.sum()) / o_flops_all.sum() * 100):.1f}%")
    
    print("=" * 70)

def analyze_load_balancing():
    """Analyze expert load balancing across layers."""
    print("\n" + "=" * 70)
    print("LOAD BALANCING ANALYSIS")
    print("=" * 70)
    
    for layer_id in sorted([k for k in elbow_topk_stats.keys() if isinstance(k, int)]):
        layer_data = elbow_topk_stats[layer_id]
        
        print(f"\nLayer {layer_id}")
        print("-" * 70)
        
        # Aggregate expert usage across all forward passes
        if "expert_usage_prefill" in layer_data and layer_data["expert_usage_prefill"]:
            # Sum across all prefill passes
            elbow_usage = torch.stack(layer_data["expert_usage_prefill"]).sum(dim=0).numpy()
            total_calls = elbow_usage.sum()
            
            print(f"\nElbow Top-K - Prefill Phase:")
            print(f"  Total expert calls: {total_calls}")
            print(f"  Expert usage distribution:")
            for expert_idx in range(len(elbow_usage)):
                usage_pct = (elbow_usage[expert_idx] / total_calls * 100) if total_calls > 0 else 0
                print(f"    Expert {expert_idx:2d}: {elbow_usage[expert_idx]:6d} calls ({usage_pct:5.2f}%)")
            
            # Calculate load balance metrics
            ideal_usage = total_calls / len(elbow_usage)
            std_dev = np.std(elbow_usage)
            coefficient_of_variation = (std_dev / ideal_usage * 100) if ideal_usage > 0 else 0
            
            print(f"  Load balance metrics:")
            print(f"    Ideal (uniform): {ideal_usage:.1f} calls per expert")
            print(f"    Std deviation: {std_dev:.2f}")
            print(f"    Coefficient of variation: {coefficient_of_variation:.2f}%")
        
        if "original_expert_usage_prefill" in layer_data and layer_data["original_expert_usage_prefill"]:
            original_usage = torch.stack(layer_data["original_expert_usage_prefill"]).sum(dim=0).numpy()
            total_calls = original_usage.sum()
            
            print(f"\nOriginal (k=8) - Prefill Phase:")
            print(f"  Total expert calls: {total_calls}")
            print(f"  Expert usage distribution:")
            for expert_idx in range(len(original_usage)):
                usage_pct = (original_usage[expert_idx] / total_calls * 100) if total_calls > 0 else 0
                print(f"    Expert {expert_idx:2d}: {original_usage[expert_idx]:6d} calls ({usage_pct:5.2f}%)")
            
            ideal_usage = total_calls / len(original_usage)
            std_dev = np.std(original_usage)
            coefficient_of_variation = (std_dev / ideal_usage * 100) if ideal_usage > 0 else 0
            
            print(f"  Load balance metrics:")
            print(f"    Ideal (uniform): {ideal_usage:.1f} calls per expert")
            print(f"    Std deviation: {std_dev:.2f}")
            print(f"    Coefficient of variation: {coefficient_of_variation:.2f}%")
    
    print("\n" + "=" * 70)

def extract_first_choice_letter(text: str) -> str | None:
    """
    Returns the first A/B/C/D letter that appears after 'Answer:' (case-insensitive),
    ignoring whitespace and punctuation. Returns None if not found.
    """
    m = re.search(r"Answer:\s*[^A-Za-z]*([A-Da-d])\b", text)
    return m.group(1).upper() if m else None

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

def format_answer(sample, benchmark):
    if benchmark == "mmlu":
        return ['A', 'B', 'C', 'D'][sample['answer']]
    if benchmark == 'arc_easy' or benchmark == 'arc_challenge':
        return sample['answerKey']
    if benchmark == 'hellaswag':
        return ['A', 'B', 'C', 'D'][int(sample['label'])]
    if benchmark == 'piqa': # 0 or 1
        return ['A', 'B'][sample['label']]
    if benchmark == 'winogrande': # '1' or '2'
        return 'A' if sample['answer'] == '1' else 'B'

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

def run_accuracy(samples, format_prompt_fn=format_mmlu_prompt, benchmark='mmlu', elbow=True):
    for sample in samples:
        prompt = format_prompt_fn(sample)
        inputs = tokenizer(prompt, return_tensors="pt")
        T = inputs["input_ids"].shape[1]
        B = inputs["input_ids"].shape[0]
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=2)
        generatedanswer = tokenizer.decode(out[0])
        singleletter = extract_first_choice_letter(generatedanswer)
        analysis = {
            'answer': format_answer(sample, benchmark),
            'generatedanswer': singleletter
        }
        if benchmark == 'mmlu':
            analysis['subject'] = sample['subject']

        if elbow:
            analysis['method'] = 'elbow'
        else:
            analysis['method'] = 'original'
        fullanalysis.append(analysis)

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



    L = len(model.model.layers)

    # attach layer_idx to each MoE block once
    for i, layer in enumerate(model.model.layers):
        layer.mlp.layer_idx = i

    elbow_topk_stats = {}

    fullanalysis = []

    test_samples_list = [ex for ex in test_samples]
    print(len(test_samples_list))


    if args.method == "elbow":
        # Test with elbow
        print("Testing ELBOW with batching...")
        use_elbow_forward()
        run_accuracy(test_samples_list, format_prompt_fn=format_prompt_fn, benchmark=args.benchmark, elbow=True)
        # save
        with open(f"{args.benchmark}_topk_stats_elbow.pkl", "wb") as f:
            pickle.dump(elbow_topk_stats, f)
        with open(f"{args.benchmark}_fullanalysis_elbow.pkl", "wb") as f:
            pickle.dump(fullanalysis, f)
    elif args.method == "original":
        # Test with original
        use_original_forward()
        print("\nTesting ORIGINAL with batching...")
        run_accuracy(test_samples_list, format_prompt_fn=format_prompt_fn, benchmark=args.benchmark, elbow=False)
        # save
        with open(f"{args.benchmark}_topk_stats_original.pkl", "wb") as f:
            pickle.dump(elbow_topk_stats, f)
        with open(f"{args.benchmark}_fullanalysis_original.pkl", "wb") as f:
            pickle.dump(fullanalysis, f)
    
    compare_k_and_flops()
    analyze_load_balancing()