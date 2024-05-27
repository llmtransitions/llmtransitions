# wrappers to load models
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, AutoTokenizer

# https://huggingface.co/docs/transformers/installation#install-with-pip
# https://huggingface.co/EleutherAI/pythia-6.9b

def my_mistral(device = 'cpu', dowload = False):
    '''
    returns the 7B mistral instruct model and its tokenizer
    '''

    if dowload:
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

        model.save_pretrained("./mistralai/Mistral-7B-Instruct-v0.1")
        tokenizer.save_pretrained("./mistralai/Mistral-7B-Instruct-v0.1")

    else:
        model = AutoModelForCausalLM.from_pretrained("./mistralai/Mistral-7B-Instruct-v0.1")
        tokenizer = AutoTokenizer.from_pretrained("./mistralai/Mistral-7B-Instruct-v0.1")

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    return model, tokenizer


def my_pythia(model_params = '70m', STEP = 143000, device = 'cpu'):
    '''
    returns a pythia model and its tokenizer
    '''

    model = GPTNeoXForCausalLM.from_pretrained(
      f"EleutherAI/pythia-{model_params}-deduped",
      revision=f"step{STEP}",
      cache_dir=f"./pythia-{model_params}-deduped/step{STEP}",
    )

    tokenizer = AutoTokenizer.from_pretrained(
      f"EleutherAI/pythia-{model_params}-deduped",
      revision=f"step{STEP}",
      cache_dir=f"./pythia-{model_params}-deduped/step{STEP}",
    )

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    return model, tokenizer
