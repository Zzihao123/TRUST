from pathlib import Path

import pandas as pd

from main.run.Args import args


def check_npy_table(df: pd.DataFrame) -> None:
    required = [args.patient_col, args.target_col, 'ph0', 'ph1', 'ph2', 'ph3']
    for col in required:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    print(f'[OK] npy_table columns: {required}')


def check_long_table(df: pd.DataFrame) -> None:
    required = [args.patient_col, args.target_col, args.view_col, args.sub_seq_col, args.image_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    print(f'[OK] long_table columns: {required}')

    view_values = set(df[args.view_col].astype(str).unique().tolist())
    missing = [v for v in ['ph0', 'ph1', 'ph2', 'ph3'] if v not in view_values]
    if missing:
        raise ValueError(f'Missing required views in `{args.view_col}`: {missing}')
    print('[OK] long_table views include ph0-ph3')


def check_paths(df: pd.DataFrame) -> None:
    root = Path(args.data_root)
    if args.data_format == 'npy_table':
        for phase in ['ph0', 'ph1', 'ph2', 'ph3']:
            missing = 0
            for rel in df[phase].astype(str).tolist():
                if not (root / rel).exists():
                    missing += 1
            print(f'[PATH] {phase}: missing {missing}/{len(df)}')
        return

    # long_table
    sample_df = df.head(min(200, len(df)))
    miss = 0
    for _, r in sample_df.iterrows():
        p = root / str(r[args.patient_col]) / str(r[args.sub_seq_col]) / (str(r[args.image_col]) + args.image_ext)
        if not p.exists():
            alt = root / str(r[args.patient_col]) / str(r[args.sub_seq_col]) / str(r[args.image_col])
            if not alt.exists():
                miss += 1
    print(f'[PATH] long_table sample check: missing {miss}/{len(sample_df)}')


def main() -> None:
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)
    print(f'[INFO] train rows={len(train_df)}, valid rows={len(valid_df)}')

    if args.data_format == 'npy_table':
        check_npy_table(train_df)
        check_npy_table(valid_df)
    else:
        check_long_table(train_df)
        check_long_table(valid_df)

    check_paths(train_df)
    print('[DONE] Data schema check passed.')


if __name__ == '__main__':
    main()
