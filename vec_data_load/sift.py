"""
http://corpus-texmex.irisa.fr/
"""

import os
import duckdb
from tqdm import tqdm
from pathlib import Path
from ivf_pq.util.fread import read_fvecs, read_ivecs


class SIFT:
    def __init__(self, data_root):
        prefix = Path(data_root).name

        self.base_path = os.path.join(data_root, f"{prefix}_base.fvecs")
        self.gt_path = os.path.join(data_root, f"{prefix}_groundtruth.ivecs")
        self.learn_path = os.path.join(data_root, f"{prefix}_learn.fvecs")
        self.query_path = os.path.join(data_root, f"{prefix}_query.fvecs")

        self.base_vecs = read_fvecs(self.base_path)
        self.gt_vecs = read_ivecs(self.gt_path)
        # self.learn_vecs = read_fvecs(self.learn_path)
        self.query_vecs = read_fvecs(self.query_path)

        self.dim = self.base_vecs.shape[-1]
        self.min_val = self.base_vecs.min()
        self.max_val = self.base_vecs.max()

    def save_db(self, db_path):
        conn = duckdb.connect(db_path)

        # save raw data
        conn.execute(f"""CREATE TABLE IF NOT EXISTS raw_data (
            id INTEGER,
            vec DOUBLE[{self.dim}],
        );""")

        for idx in tqdm(range(self.base_vecs.shape[0])):
            if conn.execute("SELECT * FROM raw_data WHERE id=?;", [idx]).fetchone():
                continue
            conn.execute(
                "INSERT INTO raw_data VALUES (?,?);",
                [idx, self.base_vecs[idx].tolist()],
            )
