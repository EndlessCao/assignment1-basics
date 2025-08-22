import torch
from nn import TransformerLM
from optim import *
from utils.data import DataLoader,Dataset
from tokenizer import Tokenizer
import pathlib
import swanlab
import json
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"
CONFIG_PATH = pathlib.Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
tokenizer = Tokenizer.from_files(
    vocab_path=os.path.join(DATA_PATH, 'tiny_vocab.json'),
    merges_path=os.path.join(DATA_PATH, 'tiny_merges.txt')
)

def do_val(model, dataloader):
    model.eval()
    total_loss = 0
    total_steps = config["train"]["val_steps"]
    with torch.no_grad():
        for i in range(total_steps):
            x, y = dataloader.get_batch("val")
            logits = model(x)
            total_loss += CrossEntropyLoss(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    prompt = "Once upon a time, "
    input_ids :torch.Tensor = tokenizer.encode(prompt, return_tensors=True)
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=100, eos_token_id=0)
    output_text = tokenizer.decode(output[0])
    print(prompt + output_text)
    return total_loss / total_steps

def train():
    save_path = pathlib.Path(config["train"]["save_path"]).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    
    swanlab.init(
        project="cs336-assignment1",
        name="train",
        config=config,
    )
    
    # load model
    model = TransformerLM(**config["model"])
    model = model.to("cuda")
    model = torch.compile(model)
    
    print(model.get_num_params())
    # load dataset
    train_data = Dataset(DATA_PATH/"tinystories_train_ids.npy")
    val_data = Dataset(DATA_PATH/"tinystories_valid_ids.npy")
    dataloader = DataLoader(train_data, val_data, batch_size=config["train"]["batch_size"], context_length=config["model"]["context_length"])
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr = config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    # scheduler
    scheduler = CosineScheduler(amax=config["optimizer"]["lr"], amin=config["optimizer"]["min_lr"], Tw=config["optimizer"]["warmup_iters"], Tc=config["optimizer"]["cosine_iters"])
    
    # train loop
    start_iter, total_iter = 0, config["train"]["train_steps"]
    process_bar = tqdm(range(start_iter, total_iter), desc="Training")
    for iteration in process_bar:
        model.train()
        x, y = dataloader.get_batch("train")
        x = x.to("cuda")
        y = y.to("cuda")
        logits = model(x)
        loss = CrossEntropyLoss(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), config["optimizer"]["clip_grad_norm"])
    
        lr = scheduler(iteration)
        optimizer.set_lr(lr)
        optimizer.step()
        
        if (iteration + 1) % config["train"]["log_interval"] == 0:
            process_bar.set_postfix({"Step": iteration, "Loss": loss.item(), "LR": lr})
            swanlab.log({"Loss": loss.item(), "LR": lr}, step=iteration)
        if (iteration + 1) % config["train"]["val_interval"] == 0:
            val_loss = do_val(model, dataloader)
            process_bar.set_postfix({"Step": iteration, "val_loss": val_loss})
            swanlab.log({"Val Loss": val_loss}, step=iteration)
        if (iteration + 1) % config["train"]["save_interval"] == 0:
            save_checkpoint(model, save_path / f"checkpoint_{iteration}.pt", optimizer, iteration)
            
    swanlab.finish()
if __name__ == "__main__":
    train()
    
    
        