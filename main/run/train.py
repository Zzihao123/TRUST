import csv
import json
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from main.data.datasets import MRIVolumeDataset
from main.model.model_my import STMRI



def _prepare_device(gpu: str) -> torch.device:
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        return torch.device('cuda')
    return torch.device('cpu')



def _build_loaders(args):
    dataset_kwargs = dict(
        data_root=args.data_root,
        num_slices=args.num_slices,
        image_size=args.image_size,
        data_format=args.data_format,
        target_col=args.target_col,
        patient_col=args.patient_col,
        view_col=args.view_col,
        sub_seq_col=args.sub_seq_col,
        image_col=args.image_col,
        image_ext=args.image_ext,
    )

    train_ds = MRIVolumeDataset(csv_path=args.train_csv, **dataset_kwargs)
    valid_ds = MRIVolumeDataset(csv_path=args.valid_csv, **dataset_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader



def _to_device(batch, device):
    images, labels, names = batch
    images = {k: [x.to(device) for x in v] for k, v in images.items()}
    labels = labels.to(device)
    return images, labels, names



def _compute_acc(logits: torch.Tensor, labels: torch.Tensor, class_num: int) -> float:
    if class_num > 1:
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean().item()

    probs = torch.sigmoid(logits.view(-1))
    preds = (probs >= 0.5).long()
    return (preds == labels.view(-1).long()).float().mean().item()



def train_one_epoch(model, loader, optimizer, device, class_num: int) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for batch in loader:
        images, labels, _ = _to_device(batch, device)
        optimizer.zero_grad()

        outputs = model(images)
        loss, _ = model.criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        final_logits = outputs[0]
        total_loss += loss.item()
        total_acc += _compute_acc(final_logits, labels, class_num)

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device, class_num: int):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    rows = []

    for batch in loader:
        images, labels, names = _to_device(batch, device)
        outputs = model(images)
        loss, _ = model.criterion(outputs, labels)

        final_logits = outputs[0]
        total_loss += loss.item()
        total_acc += _compute_acc(final_logits, labels, class_num)

        if class_num > 1:
            probs = torch.softmax(final_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = torch.max(probs, dim=1).values
        else:
            prob = torch.sigmoid(final_logits.view(-1))
            preds = (prob >= 0.5).long()
            confs = prob

        for n, gt, pd, cf in zip(names, labels.detach().cpu(), preds.detach().cpu(), confs.detach().cpu()):
            rows.append((str(n), int(gt.item()), int(pd.item()), float(cf.item())))

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n, rows



def _save_prediction_csv(rows, path: Path):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'label', 'pred', 'confidence'])
        writer.writerows(rows)



def run_train(args) -> None:
    device = _prepare_device(args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = _build_loaders(args)
    model = STMRI(kargs=args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_acc = -1.0
    best_ckpt = out_dir / 'best_model.pth'

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, args.class_num)
        val_loss, val_acc, _ = evaluate(model, valid_loader, device, args.class_num)

        print(
            f'Epoch {epoch + 1:03d} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}'
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

    metrics = {'best_val_acc': best_acc}
    with (out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f'Training finished. Best checkpoint: {best_ckpt}')


@torch.no_grad()
def run_test(args) -> None:
    if not args.weight_path:
        raise ValueError('--test requires --weight_path')

    device = _prepare_device(args.gpu)
    _, valid_loader = _build_loaders(args)

    model = STMRI(kargs=args).to(device)
    state = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(state)

    val_loss, val_acc, rows = evaluate(model, valid_loader, device, args.class_num)
    out_csv = Path(args.output_dir) / 'test_predictions.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _save_prediction_csv(rows, out_csv)
    print(f'Test result | loss={val_loss:.4f} acc={val_acc:.4f}')
    print(f'Prediction file: {out_csv}')
