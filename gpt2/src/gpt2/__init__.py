def hello() -> str:
    gpt2_showcase()
    return "Hello from gpt2!"



def gpt2_showcase():
    from transformers import GPT2LMHeadModel

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_state_dict = model_hf.state_dict()

    for k, v in gpt2_state_dict.items():
        print(k, v.shape)