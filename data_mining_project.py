"""
Data Mining Project: ทำนายความรุนแรงของอุบัติเหตุ (Severity 1-4) จาก US Accidents (Kaggle)

วิธีรัน (PowerShell, อยู่ในโฟลเดอร์โปรเจกต์ที่มี data\\US_Accidents_March23.csv):
    pip install pandas numpy scikit-learn gensim matplotlib seaborn
    python data_mining_project.py

ขั้นตอน:
  1) สุ่มข้อมูล 5% จากทั้งไฟล์ (อ่านเป็นก้อนแล้วสุ่มจากทุกก้อน)
  2) ทำความสะอาด + Feature Engineering (One-hot / Label / Frequency encoding)
  3) Word2Vec จาก Description + Pearson correlation + PCA
  4) Percentage split 80/20 + K-fold (k=5) กับ Logistic Regression / Random Forest
  5) Evaluation: Precision / Recall / F1 + Confusion matrix
  6) Ablation (ตัด Word2Vec) และโจทย์ B (ทำนายล่วงหน้า ไม่มี Description / Distance)
"""
import os
import re
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "US_Accidents_March23.csv")
SAMPLE_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
DATA_DIR = os.path.dirname(SRC) or "."
P_SAMPLE = os.path.join(DATA_DIR, "sample.csv")
P_FEAT = os.path.join(DATA_DIR, "features.csv")
P_SEL = os.path.join(DATA_DIR, "features_selected.csv")
P_PCA = os.path.join(DATA_DIR, "features_pca.csv")


def banner(t):
    print("\n" + "=" * 8 + f" {t} " + "=" * 8, flush=True)


# ------------------------------------------------------------------
# 1) สุ่มข้อมูล (stratified by position: สุ่มจากทุกก้อนของไฟล์)
# ------------------------------------------------------------------
def step_sample():
    banner("1) Sampling")
    parts = []
    for chunk in pd.read_csv(SRC, chunksize=500_000):
        parts.append(chunk.sample(frac=SAMPLE_FRAC, random_state=SEED))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(P_SAMPLE, index=False)
    print(df.shape)
    print(df["Severity"].value_counts(normalize=True))


# ------------------------------------------------------------------
# 2) Clean + Feature engineering
# ------------------------------------------------------------------
def step_features():
    banner("2) Cleaning + encoding")
    df = pd.read_csv(P_SAMPLE, low_memory=False)

    drop_cols = [
        "ID", "Source", "End_Lat", "End_Lng", "End_Time",
        "Precipitation(in)", "Wind_Chill(F)",          # missing 87-100%
        "Country", "Turning_Loop",                      # ค่าเดียวทั้งคอลัมน์
        "Zipcode", "Airport_Code", "Weather_Timestamp",
        "Nautical_Twilight", "Astronomical_Twilight",   # ซ้ำกับ Civil_Twilight
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ฟีเจอร์เวลา
    t = pd.to_datetime(df["Start_Time"].str[:19], format="%Y-%m-%d %H:%M:%S")
    df["hour"] = t.dt.hour
    df["weekday"] = t.dt.weekday
    df["month"] = t.dt.month
    df["is_weekend"] = (t.dt.weekday >= 5).astype(int)
    df = df.drop(columns=["Start_Time"])

    # เติมค่าที่ขาด
    num_cols = ["Temperature(F)", "Humidity(%)", "Pressure(in)",
                "Visibility(mi)", "Wind_Speed(mph)"]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in ["Weather_Condition", "Wind_Direction", "Timezone",
              "Sunrise_Sunset", "Civil_Twilight", "City", "County", "Street"]:
        df[c] = df[c].fillna("Unknown")

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Label encoding
    for c in ["Weather_Condition", "State"]:
        df[c + "_label"] = df[c].astype("category").cat.codes
    df = df.drop(columns=["Weather_Condition", "State"])

    # Frequency encoding
    for c in ["City", "County", "Street"]:
        freq = df[c].value_counts(normalize=True)
        df[c + "_freq"] = df[c].map(freq)
    df = df.drop(columns=["City", "County", "Street"])

    # One-hot encoding
    wd_map = {"CALM": "Calm", "VAR": "Variable", "VARIABLE": "Variable",
              "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}
    df["Wind_Direction"] = df["Wind_Direction"].str.upper().replace(wd_map)
    df = pd.get_dummies(
        df, columns=["Wind_Direction", "Timezone", "Sunrise_Sunset", "Civil_Twilight"],
        dtype=int)

    df.to_csv(P_FEAT, index=False)
    print(df.shape)
    print("missing ที่เหลือ:", int(df.drop(columns=["Description"]).isna().sum().sum()))


# ------------------------------------------------------------------
# 3) Word2Vec + Pearson + PCA
# ------------------------------------------------------------------
def step_w2v_pearson_pca():
    banner("3) Word2Vec + Pearson + PCA")
    df = pd.read_csv(P_FEAT)

    tokens = (df["Description"].fillna("").str.lower()
              .apply(lambda s: re.findall(r"[a-z0-9]+", s)))
    DIM = 50
    w2v = Word2Vec(sentences=tokens.tolist(), vector_size=DIM, window=5,
                   min_count=5, workers=4, epochs=5, seed=SEED)
    print("ขนาด vocabulary:", len(w2v.wv))
    for word in ["accident", "closed", "blocked"]:
        if word in w2v.wv:
            print(word, "->", [w for w, _ in w2v.wv.most_similar(word, topn=5)])

    def sentence_vec(words):
        words = [w for w in words if w in w2v.wv.key_to_index]
        return w2v.wv.get_mean_vector(words) if words else np.zeros(DIM)

    vecs = np.vstack(tokens.apply(sentence_vec).values)
    w2v_cols = [f"w2v_{i}" for i in range(DIM)]
    df = pd.concat([df.drop(columns=["Description"]),
                    pd.DataFrame(vecs, columns=w2v_cols)], axis=1)

    # Pearson correlation
    y = df["Severity"]
    X = df.drop(columns=["Severity"])
    base = X.drop(columns=w2v_cols)
    base = base.loc[:, base.std() > 0]
    print("\nTop 10 |r| กับ Severity:")
    print(base.corrwith(y).abs().sort_values(ascending=False).head(10))

    corr = base.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > 0.9).any()]
    print("\nตัดเพราะ correlation > 0.9:", to_drop)

    plt.figure(figsize=(16, 13))
    sns.heatmap(corr, cmap="coolwarm", xticklabels=True, yticklabels=True)
    plt.title("Pearson correlation (absolute)")
    plt.tight_layout()
    plt.savefig("corr_heatmap.png", dpi=120)
    plt.close()

    zero_var = [c for c in X.columns if c not in w2v_cols and c not in base.columns]
    X_sel = X.drop(columns=to_drop + zero_var)
    pd.concat([y, X_sel], axis=1).to_csv(P_SEL, index=False)
    print("หลัง Pearson:", X.shape[1], "->", X_sel.shape[1], "ฟีเจอร์")

    # PCA
    Z = StandardScaler().fit_transform(X_sel)
    pca = PCA(n_components=0.95, random_state=SEED)
    Z = pca.fit_transform(Z)
    print("หลัง PCA (95%):", X_sel.shape[1], "->", Z.shape[1], "components")

    plt.figure(figsize=(7, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axhline(0.95, color="r", linestyle="--")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.tight_layout()
    plt.savefig("pca_variance.png", dpi=120)
    plt.close()

    pca_df = pd.DataFrame(Z.astype("float32"),
                          columns=[f"PC{i + 1}" for i in range(Z.shape[1])])
    pd.concat([y.reset_index(drop=True), pca_df], axis=1).to_csv(P_PCA, index=False)


# ------------------------------------------------------------------
# 4-5) Model + Evaluation
# ------------------------------------------------------------------
def rf(**kw):
    return RandomForestClassifier(n_estimators=100, max_depth=20,
                                  min_samples_leaf=5, n_jobs=-1,
                                  random_state=SEED, **kw)


def logreg():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=300, class_weight="balanced"))


def step_models():
    banner("4-5) Models + Evaluation")
    df = pd.read_csv(P_SEL)
    y = df["Severity"]
    X = df.drop(columns=["Severity"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print("train:", X_train.shape, "test:", X_test.shape)

    experiments = {
        "LogReg (balanced)": (logreg(), []),
        "RF (no weight)": (rf(), []),
        "RF (balanced)": (rf(class_weight="balanced_subsample"), []),
        "RF (balanced) + PCA": (make_pipeline(
            StandardScaler(), PCA(n_components=0.95, random_state=SEED),
            rf(class_weight="balanced_subsample")), []),
        "RF (balanced) no Distance": (rf(class_weight="balanced_subsample"), ["Distance(mi)"]),
    }

    rows = []
    for name, (model, drop) in experiments.items():
        t0 = time.time()
        Xtr, Xte = X_train.drop(columns=drop), X_test.drop(columns=drop)
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0)
        rows.append({"experiment": name, "accuracy": accuracy_score(y_test, pred),
                     "precision_macro": p, "recall_macro": r, "f1_macro": f})
        print(f"\n===== {name} ({time.time() - t0:.0f} วินาที) =====", flush=True)
        print(classification_report(y_test, pred, zero_division=0, digits=3))

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, normalize="true", values_format=".2f", cmap="Blues", ax=ax)
        ax.set_title(name)
        plt.tight_layout()
        plt.savefig("cm_" + re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".png", dpi=120)
        plt.close()

    res = pd.DataFrame(rows).round(4)
    res.to_csv("results_split.csv", index=False)
    print("\n===== สรุป Percentage split =====")
    print(res.to_string(index=False))

    # K-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    kf_rows = []
    for name, model in {"LogReg (balanced)": logreg(),
                        "RF (balanced)": rf(class_weight="balanced_subsample")}.items():
        t0 = time.time()
        s = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for m in scoring:
            v = s["test_" + m]
            row[m] = f"{v.mean():.4f} ± {v.std():.4f}"
        kf_rows.append(row)
        print(f"K-fold {name} เสร็จ ({time.time() - t0:.0f} วินาที)", flush=True)
    kf = pd.DataFrame(kf_rows)
    kf.to_csv("results_kfold.csv", index=False)
    print("\n===== สรุป K-fold (5 folds) =====")
    print(kf.to_string(index=False))


# ------------------------------------------------------------------
# 6) Ablation + feature importance + โจทย์ B + คำที่สัมพันธ์กับ Severity
# ------------------------------------------------------------------
def step_ablation():
    banner("6) Ablation + importance + pre-event")
    df = pd.read_csv(P_SEL)
    y = df["Severity"]
    X = df.drop(columns=["Severity"])
    w2v = [c for c in X.columns if c.startswith("w2v_")]
    other = [c for c in X.columns if c not in w2v]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    def run(cols, name):
        m = rf(class_weight="balanced_subsample").fit(Xtr[cols], ytr)
        pred = m.predict(Xte[cols])
        _, _, f, _ = precision_recall_fscore_support(
            yte, pred, average="macro", zero_division=0)
        _, rec, _, _ = precision_recall_fscore_support(
            yte, pred, labels=[1, 2, 3, 4], zero_division=0)
        print(f"{name:34s} acc={accuracy_score(yte, pred):.4f} "
              f"macroF1={f:.4f} | recall class1-4 = {np.round(rec, 3)}", flush=True)
        return m

    full = run(list(X.columns), "all features")
    run(other, "no Word2Vec")
    run(w2v, "Word2Vec only")
    pre = [c for c in other if c != "Distance(mi)"]
    run(pre, "pre-event (no Description/Distance)")

    imp = pd.Series(full.feature_importances_, index=X.columns)
    print("\nimportance Word2Vec group:", round(imp[w2v].sum(), 3))
    print("importance other group   :", round(imp[other].sum(), 3))
    top = imp[other].sort_values(ascending=False).head(15)[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top.index, top.values)
    plt.title("Top 15 feature importance (excluding Word2Vec)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("\nTop 10:\n", top[::-1].head(10).round(4))

    # คำที่สัมพันธ์กับ Severity
    raw = pd.read_csv(P_SAMPLE, usecols=["Description", "Severity"])
    toks = raw["Description"].fillna("").str.lower().apply(
        lambda s: set(re.findall(r"[a-z]+", s)))
    cnt = Counter(w for t in toks for w in t)
    min_count = 2000 if len(raw) > 100_000 else 5
    sev = raw["Severity"].values
    rows = []
    for w, c in cnt.items():
        if c >= min_count:
            mask = toks.apply(lambda t: w in t).values
            rows.append((w, int(mask.sum()), sev[mask].mean()))
    words = pd.DataFrame(rows, columns=["word", "count", "mean_severity"])
    print("\nSeverity เฉลี่ยทั้งชุด:", round(sev.mean(), 3))
    print("\nคำที่ Severity เฉลี่ยสูงสุด:")
    print(words.sort_values("mean_severity", ascending=False).head(12).to_string(index=False))
    print("\nคำที่ Severity เฉลี่ยต่ำสุด:")
    print(words.sort_values("mean_severity").head(12).to_string(index=False))


if __name__ == "__main__":
    step_sample()
    step_features()
    step_w2v_pearson_pca()
    step_models()
    step_ablation()
    print("\nเสร็จสิ้น")
