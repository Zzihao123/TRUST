import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Arguments:
    train_csv: str
    valid_csv: str
    data_root: str
    output_dir: str
    experiment_name: str

    # Data format settings
    data_format: str
    target_col: str
    patient_col: str
    view_col: str
    sub_seq_col: str
    image_col: str
    image_ext: str

    epochs: int
    batch_size: int
    lr: float
    num_workers: int
    gpu: str
    test: bool
    weight_path: str

    class_num: int
    model_depth: int
    branch: str
    classtool: str
    mloss: bool
    enc: bool
    oadce: int

    image_size: int
    num_slices: int
    random_seed: int



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('CoPAS OpenSource (STMRI)')
    parser.add_argument('--train_csv', type=str, default='data/train.csv')
    parser.add_argument('--valid_csv', type=str, default='data/valid.csv')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--experiment_name', type=str, default='stmri_default')

    parser.add_argument('--data_format', type=str, default='npy_table', choices=['npy_table', 'long_table'])
    parser.add_argument('--target_col', type=str, default='label')
    parser.add_argument('--patient_col', type=str, default='patient_id')
    parser.add_argument('--view_col', type=str, default='view')
    parser.add_argument('--sub_seq_col', type=str, default='sub_seq')
    parser.add_argument('--image_col', type=str, default='image_id')
    parser.add_argument('--image_ext', type=str, default='.png')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--weight_path', type=str, default='')

    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--model_depth', type=int, default=34, choices=[18, 34, 50])
    parser.add_argument('--branch', type=str, default='ST', choices=['S', 'T', 'ST'])
    parser.add_argument('--classtool', type=str, default='kan', choices=['kan', 'linear'])
    parser.add_argument('--mloss', action='store_true', default=False)
    parser.add_argument('--enc', action='store_true', default=False)
    parser.add_argument('--oadce', type=int, default=4, choices=[2, 4])

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_slices', type=int, default=24)
    parser.add_argument('--random_seed', type=int, default=2024)
    return parser



def parse_args() -> Arguments:
    parser = build_parser()
    ns = parser.parse_args()

    # Auto-switch to CoPAS-like Chinese defaults for long_table
    # when user does not explicitly override these fields.
    if ns.data_format == 'long_table':
        if ns.patient_col == 'patient_id':
            ns.patient_col = '姓名'
        if ns.sub_seq_col == 'sub_seq':
            ns.sub_seq_col = '子序列'
        if ns.image_col == 'image_id':
            ns.image_col = '图序列'
        if ns.target_col == 'label':
            ns.target_col = 'label'
        if ns.view_col == 'view':
            ns.view_col = 'view'

    args = Arguments(
        train_csv=ns.train_csv,
        valid_csv=ns.valid_csv,
        data_root=ns.data_root,
        output_dir=ns.output_dir,
        experiment_name=ns.experiment_name,
        data_format=ns.data_format,
        target_col=ns.target_col,
        patient_col=ns.patient_col,
        view_col=ns.view_col,
        sub_seq_col=ns.sub_seq_col,
        image_col=ns.image_col,
        image_ext=ns.image_ext,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        lr=ns.lr,
        num_workers=ns.num_workers,
        gpu=ns.gpu,
        test=ns.test,
        weight_path=ns.weight_path,
        class_num=ns.class_num,
        model_depth=ns.model_depth,
        branch=ns.branch,
        classtool=ns.classtool,
        mloss=ns.mloss,
        enc=ns.enc,
        oadce=ns.oadce,
        image_size=ns.image_size,
        num_slices=ns.num_slices,
        random_seed=ns.random_seed,
    )
    args.output_dir = str(Path(args.output_dir) / args.experiment_name)

    # Backward-compatible aliases for legacy modules.
    setattr(args, 'ClassNum', args.class_num)
    return args


args = parse_args()
