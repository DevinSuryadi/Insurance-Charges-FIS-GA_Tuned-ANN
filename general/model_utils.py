import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MF_CENTERS = [0.2, 0.5, 0.8]
MF_SIGMAS = 0.15

N_AGE_MF = 3
N_BMI_MF = 3
N_SMOKER_MF = 2
N_RULES = N_AGE_MF * N_BMI_MF * N_SMOKER_MF

GA_POP_SIZE = 40
GA_GENS = 50
GA_ETA_C = 18
GA_ETA_M = 20
GA_ELITISM = 2
GA_CXPB = 0.9
GA_MUTPB = 0.1

ANN_LR = 0.01
ANN_EPOCHS = 100
ANN_BATCH = 32
ANN_PATIENCE = 15


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    feature_cols: list
    target_col: str


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize_smoker_column(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float)

    mapper = {
        "yes": 1.0,
        "y": 1.0,
        "true": 1.0,
        "1": 1.0,
        "smoker": 1.0,
        "no": 0.0,
        "n": 0.0,
        "false": 0.0,
        "0": 0.0,
        "non-smoker": 0.0,
    }
    return series.astype(str).str.strip().str.lower().map(mapper).fillna(0.0).astype(float)


def load_dataset(csv_path="insurance.csv", seed=RANDOM_SEED) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        rng = np.random.default_rng(seed)
        n = 1338
        age = rng.integers(18, 65, n).astype(float)
        bmi = rng.normal(30.6, 6.1, n).clip(15.96, 53.13)
        smoker = rng.choice([0, 1], n, p=[0.796, 0.204]).astype(float)
        base = 1000 + 250 * age + 50 * bmi
        mult = np.where(smoker == 1, 3.5 + 0.05 * bmi, 1.0)
        charges = base * mult * rng.lognormal(0, 0.3, n)
        pd.DataFrame({"age": age, "bmi": bmi, "smoker": smoker, "charges": charges}).to_csv(
            csv_path, index=False
        )

    raw_df = pd.read_csv(csv_path)

    if {"age", "bmi", "smoker", "charges"}.issubset(raw_df.columns):
        df = raw_df[["age", "bmi", "smoker", "charges"]].copy()
    else:
        raise ValueError("insurance.csv harus memiliki kolom age, bmi, smoker, charges")

    df["smoker"] = _normalize_smoker_column(df["smoker"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["charges"] = pd.to_numeric(df["charges"], errors="coerce")
    df.dropna(inplace=True)
    return df


def prepare_data_splits(
    df: pd.DataFrame,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    seed=RANDOM_SEED,
) -> DataBundle:
    feature_cols = ["age", "bmi", "smoker"]
    target_col = "charges"

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_all = scaler_X.fit_transform(df[feature_cols].values)
    y_all = scaler_y.fit_transform(df[[target_col]].values).ravel()

    bmi_quartile = pd.qcut(df["bmi"], 4, labels=False, duplicates="drop")

    X_train_val, X_test, y_train_val, y_test, strat_tv, _ = train_test_split(
        X_all,
        y_all,
        bmi_quartile,
        test_size=test_ratio,
        random_state=seed,
        stratify=bmi_quartile,
    )

    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_adj,
        random_state=seed,
        stratify=strat_tv,
    )

    return DataBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        target_col=target_col,
    )


class SugenoFIS:
    def __init__(
        self,
        age_centers,
        age_sigmas,
        bmi_centers,
        bmi_sigmas,
        smoker_centers,
        smoker_sigmas,
        consequents,
    ):
        self.age_c = np.array(age_centers)
        self.age_s = np.array(age_sigmas)
        self.bmi_c = np.array(bmi_centers)
        self.bmi_s = np.array(bmi_sigmas)
        self.smo_c = np.array(smoker_centers)
        self.smo_s = np.array(smoker_sigmas)
        self.conseq = np.array(consequents)

    @staticmethod
    def gaussian_mf(x, center, sigma):
        return np.exp(-0.5 * ((x - center) / (sigma + 1e-9)) ** 2)

    def fuzzify(self, age, bmi, smoker):
        mu_age = self.gaussian_mf(age, self.age_c, self.age_s)
        mu_bmi = self.gaussian_mf(bmi, self.bmi_c, self.bmi_s)
        mu_smoker = self.gaussian_mf(smoker, self.smo_c, self.smo_s)
        return mu_age, mu_bmi, mu_smoker

    def infer(self, mu_age, mu_bmi, mu_smoker):
        firing = []
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    firing.append(mu_age[i] * mu_bmi[j] * mu_smoker[k])
        return np.array(firing)

    def defuzzify(self, firing_strengths):
        total = firing_strengths.sum() + 1e-9
        return (firing_strengths * self.conseq).sum() / total

    def predict(self, X):
        preds = []
        for row in X:
            mu_a, mu_b, mu_s = self.fuzzify(row[0], row[1], row[2])
            firing = self.infer(mu_a, mu_b, mu_s)
            preds.append(self.defuzzify(firing))
        return np.array(preds)


class GATuner:
    CHROM_LEN = 34

    def __init__(
        self,
        X_train,
        y_train,
        pop_size=GA_POP_SIZE,
        generations=GA_GENS,
        eta_c=GA_ETA_C,
        eta_m=GA_ETA_M,
        elitism=GA_ELITISM,
        cxpb=GA_CXPB,
        mutpb=GA_MUTPB,
        seed=RANDOM_SEED,
    ):
        self.X = X_train
        self.y = y_train
        self.pop = pop_size
        self.gens = generations
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.elite = elitism
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.rng = np.random.default_rng(seed)
        self.history = []
        self.best_chrom = None
        self.best_fis = None

    def _decode(self, chrom):
        return dict(
            age_centers=np.clip(chrom[0:3], 0.01, 0.99),
            age_sigmas=np.clip(chrom[3:6], 0.01, 0.5),
            bmi_centers=np.clip(chrom[6:9], 0.01, 0.99),
            bmi_sigmas=np.clip(chrom[9:12], 0.01, 0.5),
            smoker_centers=np.clip(chrom[12:14], 0.0, 1.0),
            smoker_sigmas=np.clip(chrom[14:16], 0.01, 0.5),
            consequents=np.clip(chrom[16:34], 0.0, 1.0),
        )

    def evaluate_fitness(self, chrom):
        fis = SugenoFIS(**self._decode(chrom))
        pred = fis.predict(self.X)
        return np.sqrt(mean_squared_error(self.y, pred))

    def _init_population(self):
        return self.rng.uniform(0.0, 1.0, (self.pop, self.CHROM_LEN))

    def sbx_crossover(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        for i in range(self.CHROM_LEN):
            if self.rng.random() < 0.5:
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
                c1[i] = np.clip(0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i]), 0, 1)
                c2[i] = np.clip(0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i]), 0, 1)
        return c1, c2

    def polynomial_mutation(self, chrom):
        child = chrom.copy()
        for i in range(self.CHROM_LEN):
            if self.rng.random() < self.mutpb:
                u = self.rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
                child[i] = np.clip(child[i] + delta, 0, 1)
        return child

    def _tournament_select(self, population, fitnesses, k=3):
        idx = self.rng.choice(len(population), k, replace=False)
        best = idx[np.argmin(fitnesses[idx])]
        return population[best].copy()

    def run(self):
        population = self._init_population()
        fitnesses = np.array([self.evaluate_fitness(c) for c in population])
        self.history = []

        for _ in range(self.gens):
            elite_idx = np.argsort(fitnesses)[: self.elite]
            new_pop = [population[i].copy() for i in elite_idx]

            while len(new_pop) < self.pop:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                if self.rng.random() < self.cxpb:
                    c1, c2 = self.sbx_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                new_pop.append(self.polynomial_mutation(c1))
                if len(new_pop) < self.pop:
                    new_pop.append(self.polynomial_mutation(c2))

            population = np.array(new_pop)
            fitnesses = np.array([self.evaluate_fitness(c) for c in population])
            self.history.append(float(fitnesses.min()))

        best_idx = np.argmin(fitnesses)
        self.best_chrom = population[best_idx]
        self.best_fis = SugenoFIS(**self._decode(self.best_chrom))
        return self.best_chrom, self.history

    def get_best_params(self):
        if self.best_chrom is None:
            raise RuntimeError("GA belum dijalankan. Panggil run() dulu.")
        return self._decode(self.best_chrom)


class NeuroFuzzyNet(nn.Module):
    def __init__(self, n_age=3, n_bmi=3, n_smo=2):
        super().__init__()
        self.n_age = n_age
        self.n_bmi = n_bmi
        self.n_smo = n_smo
        self.n_rules = n_age * n_bmi * n_smo

        self.age_c = nn.Parameter(torch.tensor([0.2, 0.5, 0.8]))
        self.age_ls = nn.Parameter(torch.full((n_age,), -1.9))

        self.bmi_c = nn.Parameter(torch.tensor([0.2, 0.5, 0.8]))
        self.bmi_ls = nn.Parameter(torch.full((n_bmi,), -1.9))

        self.smo_c = nn.Parameter(torch.tensor([0.1, 0.9]))
        self.smo_ls = nn.Parameter(torch.full((n_smo,), -1.9))

        self.consequents = nn.Parameter(torch.rand(self.n_rules))

    def _gaussian(self, x, centers, log_sigmas):
        sigmas = torch.exp(log_sigmas).clamp(min=1e-3)
        x_exp = x.unsqueeze(1)
        return torch.exp(-0.5 * ((x_exp - centers) / sigmas) ** 2)

    def forward(self, X):
        mu_age = self._gaussian(X[:, 0], self.age_c, self.age_ls)
        mu_bmi = self._gaussian(X[:, 1], self.bmi_c, self.bmi_ls)
        mu_smo = self._gaussian(X[:, 2], self.smo_c, self.smo_ls)

        rules = []
        for i in range(self.n_age):
            for j in range(self.n_bmi):
                for k in range(self.n_smo):
                    rules.append(mu_age[:, i] * mu_bmi[:, j] * mu_smo[:, k])
        firing = torch.stack(rules, dim=1)

        w_sum = firing.sum(dim=1, keepdim=True) + 1e-9
        output = (firing * self.consequents).sum(dim=1) / w_sum.squeeze(1)
        return output.clamp(0, 1)


def compute_metrics(y_true_norm, y_pred_norm, scaler_y):
    r2_n = r2_score(y_true_norm, y_pred_norm)
    rmse_n = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))
    mae_n = mean_absolute_error(y_true_norm, y_pred_norm)

    y_true_d = scaler_y.inverse_transform(y_true_norm.reshape(-1, 1)).ravel()
    y_pred_d = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()

    r2_d = r2_score(y_true_d, y_pred_d)
    rmse_d = np.sqrt(mean_squared_error(y_true_d, y_pred_d))
    mae_d = mean_absolute_error(y_true_d, y_pred_d)

    return {
        "R2_norm": float(r2_n),
        "RMSE_norm": float(rmse_n),
        "MAE_norm": float(mae_n),
        "R2_usd": float(r2_d),
        "RMSE_usd": float(rmse_d),
        "MAE_usd": float(mae_d),
    }


def get_manual_fis():
    manual_conseq = np.array(
        [
            0.10,
            0.35,
            0.20,
            0.50,
            0.30,
            0.65,
            0.25,
            0.55,
            0.40,
            0.70,
            0.50,
            0.80,
            0.45,
            0.75,
            0.60,
            0.85,
            0.70,
            0.95,
        ]
    )

    return SugenoFIS(
        age_centers=MF_CENTERS,
        age_sigmas=[MF_SIGMAS] * 3,
        bmi_centers=MF_CENTERS,
        bmi_sigmas=[MF_SIGMAS] * 3,
        smoker_centers=[0.2, 0.8],
        smoker_sigmas=[MF_SIGMAS] * 2,
        consequents=manual_conseq,
    )


def train_ann(
    X_train,
    y_train,
    X_val,
    y_val,
    lr=ANN_LR,
    epochs=ANN_EPOCHS,
    batch_size=ANN_BATCH,
    patience=ANN_PATIENCE,
    seed=RANDOM_SEED,
):
    Xt = torch.FloatTensor(X_train)
    yt = torch.FloatTensor(y_train)
    Xv = torch.FloatTensor(X_val)
    yv = torch.FloatTensor(y_val)

    train_ds = TensorDataset(Xt, yt)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = NeuroFuzzyNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loss = []
    val_loss = []
    best_val_loss = float("inf")
    patience_cnt = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        batch_losses = []
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        current_train_loss = float(np.mean(batch_losses))
        model.eval()
        with torch.no_grad():
            current_val_loss = float(criterion(model(Xv), yv).item())

        train_loss.append(current_train_loss)
        val_loss.append(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    model.load_state_dict(best_state)
    return model, train_loss, val_loss


def run_full_pipeline(csv_path="insurance.csv"):
    set_seed(RANDOM_SEED)

    df = load_dataset(csv_path=csv_path, seed=RANDOM_SEED)
    data = prepare_data_splits(df, seed=RANDOM_SEED)

    manual_fis = get_manual_fis()
    y_test_manual = manual_fis.predict(data.X_test)
    metrics_manual = compute_metrics(data.y_test, y_test_manual, data.scaler_y)

    ga_tuner = GATuner(X_train=data.X_train, y_train=data.y_train)
    ga_tuner.run()
    ga_fis = ga_tuner.best_fis
    ga_params = ga_tuner.get_best_params()
    y_test_ga = ga_fis.predict(data.X_test)
    metrics_ga = compute_metrics(data.y_test, y_test_ga, data.scaler_y)

    ann_model, ann_train_loss, ann_val_loss = train_ann(
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
    )
    ann_model.eval()
    with torch.no_grad():
        y_test_ann = ann_model(torch.FloatTensor(data.X_test)).numpy()
    metrics_ann = compute_metrics(data.y_test, y_test_ann, data.scaler_y)

    return {
        "data": data,
        "df_shape": tuple(df.shape),
        "manual_fis": manual_fis,
        "ga_fis": ga_fis,
        "ga_params": ga_params,
        "ann_model": ann_model,
        "ann_train_loss": ann_train_loss,
        "ann_val_loss": ann_val_loss,
        "metrics": {
            "manual": metrics_manual,
            "ga": metrics_ga,
            "ann": metrics_ann,
        },
    }


def serializable_fis_params(fis: SugenoFIS):
    return {
        "age_centers": fis.age_c.tolist(),
        "age_sigmas": fis.age_s.tolist(),
        "bmi_centers": fis.bmi_c.tolist(),
        "bmi_sigmas": fis.bmi_s.tolist(),
        "smoker_centers": fis.smo_c.tolist(),
        "smoker_sigmas": fis.smo_s.tolist(),
        "consequents": fis.conseq.tolist(),
    }


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
