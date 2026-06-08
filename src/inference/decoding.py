import torch
from jaxtyping import Float, Int

from model.transformer import TransformerLM, temp_softmax


@torch.inference_mode()
def decoding(
    model: TransformerLM,
    prompt: Int[torch.Tensor, "batch_size seq_len"],
    eos_token_id: int,
    max_completion_length: int,
    temperature: float,
    p: float,
)-> Int[torch.Tensor, "batch_size completion_length"]:
    """ Decoding a batcg of prompts using temperature scaling and nucleus sampling.

    Args:
        model: The language model used for decoding.
        prompt: A batch of input prompts with shape (batch_size, seq_len).
        eos_token_id: The token ID representing the end of a sequence.
        max_completion_length: The maximum length of the generated completion.
        temperature: The temperature for scaling the logits before sampling.
        p: The cumulative probability threshold for nucleus sampling.

    Returns:
        A batch of generated completions with shape (batch_size, completion_length).
    """
    if p > 1.0 or p <= 0.0:
        raise ValueError(f"decoding(): an illegal p:{p}.")
    if temperature <= 0.0:
        raise ValueError(f"decoding(): an illegal temperature:{temperature}.")
    if max_completion_length < 0:
        raise ValueError(f"decoding(): an illegal max_completion_length:{max_completion_length}.")
    if prompt.shape[-1] <= 0:
        raise ValueError(f"decoding(): an illegal prompt:P{prompt}")
    if prompt.shape[-1] > model.context_length:
        raise ValueError(f"decoding(): the length of prompt exceeds the context length of model.")
    if prompt.ndim != 2:
        raise ValueError(f"decoding(): an illegal prompt with shape {prompt.shape}.")
    
    original_prompt_length = prompt.shape[-1]
    max_length = model.context_length - original_prompt_length
    finished = torch.zeros(
        (prompt.shape[0], 1),
        dtype=torch.bool,
        device=prompt.device,
    )

    for _ in range(min(max_completion_length,max_length)):
        new_token = decode_a_token(
            model=model,
            prompt=prompt,
            temperature=temperature,
            p=p,
            )
        
        new_token = torch.where(
            finished,
            torch.full_like(new_token, eos_token_id),
            new_token,
        )
        
        prompt = torch.cat([prompt, new_token], dim=-1)
        finished |= new_token.eq(eos_token_id)

        if finished.all():
            break

    return  prompt[..., original_prompt_length:]


def decode_a_token(
    model: TransformerLM,
    prompt: Int[torch.Tensor, "batch_size seq_len"],
    temperature: float,
    p: float,
) -> Int[torch.Tensor,"batch_size 1"]:
    """Decoding a single token index from prompt.

    Args:   
        model: The language model used for decoding.
        prompt: A batch of input prompts with shape (batch_size, seq_len).
        temperature: The temperature for scaling the logits before sampling.
        p: The cumulative probability threshold for nucleus sampling.

    Returns:
        A batch of generated token indices with shape (batch_size, 1).
    """

    logits = model(prompt)
    probability = last_elem_temp_scaling_softmax(temperature,logits)

    return top_p_sampling(probability, p)


def last_elem_temp_scaling_softmax(
    temperature: float, 
    logits: Float[torch.Tensor, "batch_size seq_len vocab_size"],
) -> Float[torch.Tensor,"batch_size vocab_size"]:
    """Temperature scaling softmax for last element in logits
    """
    logits = logits[..., -1, :]
    return  temp_softmax(x=logits, dim=-1, temperature=temperature)


def top_p_sampling(
    prob: Float[torch.Tensor, "batch_size vocab_size"],
    p: float,
) -> Int[torch.Tensor, "batch_size 1"]:
    """Sample token indices using nucleus sampling.

    Args:
        prob: Probability distribution over the vocabulary.
        p: Cumulative probability threshold in the interval (0, 1].

    Returns:
        Sampled token indices.
    """
    sorted_prob, indices = torch.sort(prob, dim=-1, descending=True)
    cum_prob = torch.cumsum(sorted_prob, dim=-1)

    # Remove the token whose accumulated probability surpass p.
    # Note: keep the first token with which the accumulated probabliry surpass p.
    removed_mask = cum_prob > p
    removed_mask[..., 1:] = removed_mask[..., :-1].clone()
    removed_mask[..., 0] =False

    sorted_prob = sorted_prob.masked_fill(removed_mask, 0.0)
    sorted_prob /= torch.sum(sorted_prob, dim=-1, keepdim=True)

    sampled_position = torch.multinomial(sorted_prob, num_samples=1)
    sampled_token = torch.gather(
        indices,
        dim=-1,
        index=sampled_position
    )

    return sampled_token
