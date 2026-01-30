import os
import timeit
import torch
import random
import wandb

from tokenizer import Tokenizer, train_bpt_tokenizer
from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH
from optimizer import SGD

special_tokens = ['tok', 'start', 'end']

def merge_dicts():
    new_dict = {**{2: 'a', 3: 'b'}, **dict(enumerate(special_tokens, 256))}

def encode_bytes():
    bytes_obj = 'ćtr'.encode('utf-8')
    tuple_obj = tuple(bytes_obj)
    print(tuple_obj)

def compare_tuples():
    tup1 = (b're', b's')
    tup2 = (b' s', b't')
    print('re' > 's')  # should be false, it goes O P _R_ _S_, 's'>'r'
    print('re' > ' s')  # should be true, it goes ' ' ... 'r', 'r' > ' '

    tup1 = (b' c', b'om')
    tup2 = (b't', b'h')
    tup1 = ((32, 99), (111, 109))
    tup2 = ((116,), (104,))
    print(max(tup1, tup2))  # should be false since space comes before t.

    # Hence tiebreaking should be fine

def tokenizer_testing():
    vocab, merges = train_bpt_tokenizer("testfile.txt", 300, ["<|end_of_text|>"])
    print(f"{merges=}")
    print(f"{vocab=}")

    with open("testfile.txt", 'r') as f:
        text = f.read()

    this_tokenizer = Tokenizer(vocab, merges)
    encoded = this_tokenizer.encode(text)
    print(encoded)
    decoded = this_tokenizer.decode(encoded)
    print(decoded)


def tokenizer_more_testing():
    my_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path="../tests/fixtures/gpt2_vocab.json",
        merges_path="../tests/fixtures/gpt2_merges.txt",
    )
    test_string =  "Héllò hôw are ü? 🙃"
    encoded_ids = my_tokenizer.encode(test_string)
    print(encoded_ids)
    decoded_string = my_tokenizer.decode(encoded_ids)
    print(test_string, decoded_string)
    assert test_string == decoded_string

def tokenizer_special_testing():
    my_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = my_tokenizer.encode(test_string)
    print(encoded_ids)
    tokenized_string = [my_tokenizer.decode([x]) for x in encoded_ids]
    print(tokenized_string)
    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count("<|endoftext|>") == 3

    decoded_string = my_tokenizer.decode(encoded_ids)
    assert test_string == decoded_string

def tokenizer_little_tiktoken_test():
    my_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    ids = my_tokenizer.encode(test_string)

    assert my_tokenizer.decode(ids) == test_string


def summing_test():
    a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    print(a.size())
    print(a.pow(2)) # [3, 2]
    rms = a.pow(2).sum(-1, keepdim=True) / 2 + 0.1
    print(rms) # [3, 1]
    print(a / rms)


def power_test():
    a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    print(torch.pow(2, a))

def roll_test():
    a = torch.Tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    print(a)
    print(torch.roll(a, 1))

def max_text():
    a = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    print(torch.max(a, dim=0, keepdim=True).values)

def indexing_test():
    a = torch.Tensor([
        [0, 1],
        [2, 3],
        [0, 1],
        [2, 3]])
    print(a[[0, 2]])

def optimizer_test():
    lrs = [1, 1e1, 1e2, 1e3]

    for lr in lrs:
        print(f"For {lr=}, in 10 iterations:")

        weights = torch.nn.Parameter(5* torch.randn(10, 10))
        opt = SGD([weights], lr=lr)

        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(loss.cpu().item())
            loss.backward()
            opt.step()

def speed_testing_where(x: torch.Tensor):
    z = torch.where(x <= 200000, 0, x)

def speed_testing_mask(x: torch.Tensor):
    y = (x <= 200000) * x

def speed_testing():
    x = torch.arange(1000000)
    time_where = timeit.timeit(lambda: speed_testing_where(x), number=10000)
    time_mask = timeit.timeit(lambda: speed_testing_mask(x), number=10000)

    print(f"{time_where=}")
    print(f"{time_mask=}")

def sort_testing():
    x = torch.arange(10000, 1000000)
    x_sorted = x.sort(descending=True)

    print(x_sorted)
    print(type(x_sorted))

sort_testing()

def wandb_testing():

    wandb.login()

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="emibrahimovic-mit",
        # Set the wandb project where this run will be logged.
        project="my-awesome-project",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )

    # Simulate training.
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # Log metrics to wandb.
        run.log({"acc": acc, "loss": loss})

    # Finish the run and upload any remaining data.
    run.finish()

# wandb_testing()


