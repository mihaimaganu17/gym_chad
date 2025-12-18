from gpt2.train import GPT

def hello() -> str:
    gpt2_showcase()
    model = GPT.from_pretrained('gpt2')
    print("didn't crash")
    
    # Sampling from the HF GPT2
    #for example in gpt2_sample():
    #    print(example)
    return "Hello from gpt2!"



def gpt2_showcase():
    from transformers import GPT2LMHeadModel

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_state_dict = model_hf.state_dict()

    for k, v in gpt2_state_dict.items():
        print(k, v.shape)


def gpt2_sample():
    from transformers import pipeline, set_seed
    generator = pipeline('text-generation', model='gpt2')
    set_seed(1337)
    examples = generator("Hello, I'm a langauge model", max_new_tokens=30, num_return_sequences=5)
    return examples