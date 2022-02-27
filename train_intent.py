from cmath import inf
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from model import SeqClassifier

import torch
from tqdm import trange
import tqdm

from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets}
    dataloader = {split : torch.utils.data.DataLoader(dataset=datasets[split], batch_size=args.batch_size, shuffle=False, collate_fn=datasets[split].collate_fn) for split in SPLITS}

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, len(intent2idx))
    device = args.device
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr) # could specify some more arguments
    criterion = torch.nn.CrossEntropyLoss()


    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    max_eval_acc = 0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_acc = 0
        train_loss = 0
        for _, batch in enumerate(tqdm.tqdm(dataloader[TRAIN])):
            text = batch["text"].to(device)
            intent_truth = batch["intent"].to(device)
            optimizer.zero_grad()

            out = model(text)
            loss = criterion(out, intent_truth)
            loss.backward()
            optimizer.step()

            _, prediction = torch.max(out, 1)
            train_acc += torch.sum(prediction==intent_truth)
            train_loss += loss

        train_acc /= len(dataloader[TRAIN])
        train_loss /= len(dataloader[TRAIN])
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            model.eval()
            eval_acc = 0
            eval_loss = 0
            for _, batch in enumerate(tqdm.tqdm(dataloader[DEV])):
                text = batch["text"].to(device)
                intent_truth = batch["intent"].to(device)

                out = model(text)
                loss = criterion(out, intent_truth)

                _, prediction = torch.max(out, 1)
                eval_acc += torch.sum(prediction==intent_truth)
                eval_loss += loss

            eval_acc /= len(dataloader[DEV])
            eval_loss /= len(dataloader[DEV])
            print(f"Epoch {epoch}:\n    Train: Accuracy={train_acc}, Loss={train_loss}\n    Eval: Accuracy={eval_acc}, Loss={eval_loss}")
            if eval_acc >= max_eval_acc:
                max_eval_acc = eval_acc
                torch.save(model.state_dict(), args.ckpt_dir / "model.ckpt")
                print("model saved")
        pass

    # TODO: Inference on test set
    test_path = args.data_dir / "test.json"
    test_data = json.loads(test_path.read_text())
    test_dataset = SeqClsDataset(test_data, vocab, intent2idx, args.max_len)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    model.eval()

    with open("./prediction/intent.csv", "w") as outfile:
        outfile.write("id,intent\n")
        with torch.no_grad():
            for _, batch in enumerate(tqdm.tqdm(test_dataloader)):
                text = batch["text"].to(device)
                out = model(text)
                _, prediction = torch.max(out, 1)
                prediction = prediction.tolist()
                zipped = zip(batch["id"],prediction)
                for line in zipped:
                    outfile.write(f"{line[0]},{line[1]}\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
