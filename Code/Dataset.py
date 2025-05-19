from torch.utils.data import Dataset
import numpy as np
import Config
import sqlite3
import pickle


class MEGA_DRP_Dataset(Dataset):

    def __init__(self, mode, k_fold=None, cell=None, drug=None):
        super(MEGA_DRP_Dataset).__init__()

        self.mode = mode
        self.blind = Config.blind
        self.k_fold = k_fold

        if self.mode in ["train", "valid", "test"]:

            dataset = Config.datasets[0]
            self.conn = sqlite3.connect(dataset)

            if self.blind == "mix":

                sql = "SELECT id, cell_id, drug_id, LnIC50 FROM response"
                result = self.conn.execute(sql)
                rows = result.fetchall()

                rows = np.array(rows)
                rng = np.random.default_rng(Config.seed)
                rows = rng.permutation(rows)

                a = round(self.k_fold * 0.2 * len(rows))
                b = round((self.k_fold + 1) * 0.2 * len(rows))

                train_set = rows[:a].tolist() + rows[b:].tolist()
                test_set = rows[a:b].tolist()

                if self.mode == "train":
                    self.data = train_set[:round(len(train_set) * 0.875)]
                if self.mode == "valid":
                    self.data = train_set[round(len(train_set) * 0.875):]
                if self.mode == "test":
                    self.data = test_set

            if self.blind == "cb":

                with open(f"../Data/Blind_Division/c_train_{self.k_fold}.txt", "r") as input_file:
                    lines = input_file.readlines()
                train_cell_list = [line.strip() for line in lines]

                sql = "SELECT id, cell_id, drug_id, LnIC50 FROM response WHERE cell_id IN ({})"
                result = self.conn.execute(sql.format(",".join("?" for _ in train_cell_list)), train_cell_list)
                rows = result.fetchall()

                rows = np.array(rows)
                rng = np.random.default_rng(Config.seed)
                rows = rng.permutation(rows)

                train_set = rows.tolist()

                with open(f"../Data/Blind_Division/c_test_{self.k_fold}.txt", "r") as input_file:
                    lines = input_file.readlines()
                test_cell_list = [line.strip() for line in lines]

                sql = "SELECT id, cell_id, drug_id, LnIC50 FROM response WHERE cell_id IN ({})"
                result = self.conn.execute(sql.format(",".join("?" for _ in test_cell_list)), test_cell_list)
                rows = result.fetchall()
                test_set = rows

                if self.mode == "train":
                    self.data = train_set[:round(len(train_set) * 0.875)]
                if self.mode == "valid":
                    self.data = train_set[round(len(train_set) * 0.875):]
                if self.mode == "test":
                    self.data = test_set

            if self.blind == "db":

                with open(f"../Data/Blind_Division/d_train_{self.k_fold}.txt", "r") as input_file:
                    lines = input_file.readlines()
                train_drug_list = [line.strip() for line in lines]

                sql = "SELECT id, cell_id, drug_id, LnIC50 FROM response WHERE drug_id IN ({})"
                result = self.conn.execute(sql.format(",".join("?" for _ in train_drug_list)), train_drug_list)
                rows = result.fetchall()

                rows = np.array(rows)
                rng = np.random.default_rng(Config.seed)
                rows = rng.permutation(rows)

                train_set = rows.tolist()

                with open(f"../Data/Blind_Division/d_test_{self.k_fold}.txt", "r") as input_file:
                    lines = input_file.readlines()
                test_drug_list = [line.strip() for line in lines]

                sql = "SELECT id, cell_id, drug_id, LnIC50 FROM response WHERE drug_id IN ({})"
                result = self.conn.execute(sql.format(",".join("?" for _ in test_drug_list)), test_drug_list)
                rows = result.fetchall()
                test_set = rows

                if self.mode == "train":
                    self.data = train_set[:round(len(train_set) * 0.875)]
                if self.mode == "valid":
                    self.data = train_set[round(len(train_set) * 0.875):]
                if self.mode == "test":
                    self.data = test_set

            with open(f"../Data/G{Config.n_gene}/g_gdsc.txt", "r") as file:
                lines = file.read().splitlines()
                self.gene_indexes = list(map(int, lines))

        if self.mode == "test_tcga":

            dataset = Config.datasets[1]
            self.conn = sqlite3.connect(dataset)

            sql = "SELECT id, cell_id, drug_id FROM response"
            result = self.conn.execute(sql)
            rows = result.fetchall()
            self.data = rows

            with open(f"../Data/G{Config.n_gene}/g_tcga.txt", "r") as file:
                lines = file.read().splitlines()
                self.gene_indexes = list(map(int, lines))

        if self.mode == "test_sc":

            dataset = Config.datasets[2]
            self.conn = sqlite3.connect(dataset)

            sql = "SELECT id, cell_id, drug_id, LnIC50 FROM response"
            result = self.conn.execute(sql)
            rows = result.fetchall()
            self.data = rows

            with open(f"../Data/G{Config.n_gene}/g_sc.txt", "r") as file:
                lines = file.read().splitlines()
                self.gene_indexes = list(map(int, lines))

        if self.mode == "test_pdx":

            dataset = Config.datasets[3]
            self.conn = sqlite3.connect(dataset)

            sql = "SELECT id, cell_id, drug_id FROM response"
            result = self.conn.execute(sql)
            rows = result.fetchall()
            self.data = rows

            with open(f"../Data/G{Config.n_gene}/g_pdx.txt", "r") as file:
                lines = file.read().splitlines()
                self.gene_indexes = list(map(int, lines))

        if self.mode == "interpret":

            dataset = Config.datasets[0]
            self.conn = sqlite3.connect(dataset)

            sql = "SELECT id, cell_id, drug_id FROM response"
            params = []

            if cell and drug:
                sql += " WHERE cell LIKE ? AND drug LIKE ?"
                params.extend([f"%{cell}%", f"%{drug}%"])
            elif cell:
                sql += " WHERE cell LIKE ?"
                params.append(f"%{cell}%")
            elif drug:
                sql += " WHERE drug LIKE ?"
                params.append(f"%{drug}%")
            result = self.conn.execute(sql, params)
            rows = result.fetchall()
            self.data = rows

            with open(f"../Data/G{Config.n_gene}/g_gdsc.txt", "r") as file:
                lines = file.read().splitlines()
                self.gene_indexes = list(map(int, lines))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        response_id = int(self.data[idx][0])
        cell_id = int(self.data[idx][1])
        drug_id = int(self.data[idx][2])

        result = self.conn.execute("SELECT exp, mut, cnv FROM cell WHERE id = ?", (cell_id,))
        row = result.fetchone()
        exp = pickle.loads(row[0])
        mut = pickle.loads(row[1])
        cnv = pickle.loads(row[2])

        gene_indexes = self.gene_indexes

        result = self.conn.execute("SELECT graph FROM drug WHERE id = ?", (drug_id,))
        row = result.fetchone()
        drug_graph = pickle.loads(row[0])
        adj_mat = drug_graph[0]
        v_feature = drug_graph[1]

        label = self.data[idx][3] if self.mode in ["train", "valid", "test", "test_sc"] else None

        return response_id, cell_id, drug_id, gene_indexes, exp, mut, cnv, adj_mat, v_feature, label
