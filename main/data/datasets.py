from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MRIVolumeDataset(Dataset):
    """STMRI dataset with two formats.

    Format A: npy_table
      Required columns: patient_id, label, ph0, ph1, ph2, ph3
      Optional columns: DWI, T1, T2
      Each phase cell is a relative path to a .npy volume.

    Format B: long_table
      Required columns: patient_col, target_col, view_col, sub_seq_col, image_col
      The loader groups rows by patient and builds phase volumes from image slices.
      Views must include ph0, ph1, ph2, ph3.
    """

    REQUIRED_PHASES = ['ph0', 'ph1', 'ph2', 'ph3']
    OPTIONAL_AUX = ['DWI', 'T1', 'T2']

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        num_slices: int = 24,
        image_size: int = 224,
        data_format: str = 'npy_table',
        target_col: str = 'label',
        patient_col: str = 'patient_id',
        view_col: str = 'view',
        sub_seq_col: str = 'sub_seq',
        image_col: str = 'image_id',
        image_ext: str = '.png',
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.num_slices = num_slices
        self.image_size = image_size
        self.data_format = data_format

        self.target_col = target_col
        self.patient_col = patient_col
        self.view_col = view_col
        self.sub_seq_col = sub_seq_col
        self.image_col = image_col
        self.image_ext = image_ext

        if self.data_format == 'npy_table':
            self._init_npy_table()
        elif self.data_format == 'long_table':
            self._init_long_table()
        else:
            raise ValueError(f'Unsupported data_format: {self.data_format}')

    def _init_npy_table(self):
        for col in [self.patient_col, self.target_col, *self.REQUIRED_PHASES]:
            if col not in self.df.columns:
                raise ValueError(f'Missing required column: {col}')

    def _init_long_table(self):
        for col in [self.patient_col, self.target_col, self.view_col, self.sub_seq_col, self.image_col]:
            if col not in self.df.columns:
                raise ValueError(f'Missing required column: {col}')
        self.df = self.df.drop_duplicates(subset=[self.patient_col, self.view_col, self.sub_seq_col, self.image_col])
        self.patient_ids = self.df[self.patient_col].astype(str).unique().tolist()

    def __len__(self) -> int:
        if self.data_format == 'long_table':
            return len(self.patient_ids)
        return len(self.df)

    def _resize_5d(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=(self.num_slices, self.image_size, self.image_size),
            mode='trilinear',
            align_corners=False,
        )
        return x.squeeze(0)

    def _load_npy_volume(self, rel_path: str) -> torch.Tensor:
        arr = np.load(self.data_root / rel_path)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f'Invalid volume shape: {arr.shape} for {rel_path}')
        x = torch.from_numpy(arr).float()
        return self._resize_5d(x)

    def _load_slice_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert('L').resize((self.image_size, self.image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def _build_volume_from_rows(self, rows: pd.DataFrame) -> torch.Tensor:
        # sort by image id as sequence proxy
        rows = rows.copy()
        rows[self.image_col] = rows[self.image_col].astype(str)
        rows = rows.sort_values(by=self.image_col)

        slices = []
        for _, r in rows.iterrows():
            patient = str(r[self.patient_col])
            sub_seq = str(r[self.sub_seq_col])
            image_id = str(r[self.image_col])
            path = self.data_root / patient / sub_seq / (image_id + self.image_ext)
            if not path.exists():
                # allow image_id already containing extension
                alt = self.data_root / patient / sub_seq / image_id
                if alt.exists():
                    path = alt
                else:
                    continue
            slices.append(self._load_slice_image(path))

        if not slices:
            vol = torch.zeros((1, self.num_slices, self.image_size, self.image_size), dtype=torch.float32)
            return vol

        arr = np.stack(slices, axis=0)  # [D, H, W]
        x = torch.from_numpy(arr).unsqueeze(0)  # [1, D, H, W]
        return self._resize_5d(x)

    def _get_item_npy_table(self, index: int):
        row = self.df.iloc[index]

        ph_list: List[torch.Tensor] = [self._load_npy_volume(row[k]) for k in self.REQUIRED_PHASES]
        sample: Dict[str, List[torch.Tensor]] = {'ph': ph_list}

        for key in self.OPTIONAL_AUX:
            if key in self.df.columns and isinstance(row.get(key, ''), str) and row.get(key, ''):
                sample[key] = [self._load_npy_volume(row[key])]

        label = torch.tensor(int(row[self.target_col]), dtype=torch.long)
        patient_id = str(row[self.patient_col])
        return sample, label, patient_id

    def _get_item_long_table(self, index: int):
        patient_id = self.patient_ids[index]
        patient_rows = self.df[self.df[self.patient_col].astype(str) == patient_id]

        sample: Dict[str, List[torch.Tensor]] = {'ph': []}

        for view_name in self.REQUIRED_PHASES:
            view_rows = patient_rows[patient_rows[self.view_col].astype(str) == view_name]
            sample['ph'].append(self._build_volume_from_rows(view_rows))

        for aux_name in self.OPTIONAL_AUX:
            aux_rows = patient_rows[patient_rows[self.view_col].astype(str) == aux_name]
            if len(aux_rows) > 0:
                sample[aux_name] = [self._build_volume_from_rows(aux_rows)]

        label = int(patient_rows.iloc[0][self.target_col])
        return sample, torch.tensor(label, dtype=torch.long), patient_id

    def __getitem__(self, index: int):
        if self.data_format == 'long_table':
            return self._get_item_long_table(index)
        return self._get_item_npy_table(index)
