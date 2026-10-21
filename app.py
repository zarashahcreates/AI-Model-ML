"""ML Pipeline Doctor AI - every number shown is computed from the data or metrics you provide."""
import io, json, os, re, traceback, warnings
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ML Pipeline Doctor AI", page_icon="🩺", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] {font-family:'IBM Plex Sans',sans-serif;}
.hero {background:linear-gradient(120deg,#0f3d3e,#14675f);color:#f2fbf9;padding:1.4rem 1.6rem;border-radius:10px;margin-bottom:1rem}
.hero h1 {margin:0;font-size:1.8rem;color:#fff} .hero p {margin:.3rem 0 0;opacity:.9}
.finding {border-left:5px solid var(--c);padding:.6rem 1rem;margin:.6rem 0;background:rgba(127,127,127,.08);border-radius:4px}
@media (max-width:640px){.hero h1{font-size:1.3rem}}
</style>""", unsafe_allow_html=True)

COLORS = {"Critical": "#d64545", "Warning": "#e08a1e", "Info": "#2f7fc1", "OK": "#2a9d6f"}
ICON = {"Critical": "🔴", "Warning": "🟠", "Info": "🔵", "OK": "🟢"}
ORDER = {"Critical": 0, "Warning": 1, "Info": 2, "OK": 3}
LEAK_WORDS = ("target", "label", "outcome", "result", "after", "future", "churned", "leak", "response")


def F(sev, title, evidence, why, causes, steps, code=""):
    return dict(sev=sev, title=title, evidence=evidence, why=why, causes=causes, steps=steps, code=code)


# ----------------------------------------------------------------------------- data
def make_sample(name, seed=42):
    rng = np.random.default_rng(seed)
    def clf(n, f, inf, w=None, flip=0.03):
        X, y = make_classification(n, f, n_informative=inf, n_redundant=0, weights=w, flip_y=flip, class_sep=1.0, random_state=seed)
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(f)]), y
    if name == "Healthy classification":
        X, y = clf(1500, 10, 6)
    elif name == "Data leakage":
        X, y = clf(1500, 8, 5)
        X["post_outcome_score"] = y * 2.0 + rng.normal(0, 0.15, len(y))
        X["customer_id"] = np.arange(len(y))
    elif name == "Imbalance + missing values":
        X, y = clf(2000, 8, 5, w=[0.96, 0.04], flip=0.01)
        for c in ["feature_1", "feature_3", "feature_5"]:
            X.loc[rng.random(len(X)) < 0.18, c] = np.nan
        X["region"] = rng.choice(["north", "south", "east", "west"], len(X))
        dup = X.sample(100, random_state=1)
        X = pd.concat([X, dup], ignore_index=True); y = np.concatenate([y, y[dup.index]])
    elif name == "Overfitting (few rows, many features)":
        X, y = clf(120, 60, 4, flip=0.1)
    elif name == "Underfitting (non-linear regression)":
        X = pd.DataFrame(rng.uniform(-3, 3, (1500, 4)), columns=[f"x{i}" for i in range(4)])
        y = np.sin(X.x0 * 2) * 3 + (X.x1 ** 2) + rng.normal(0, 0.3, 1500)
        df = X.copy(); df["target"] = y.values; return df
    df = X.copy(); df["target"] = y
    return df


def infer_task(y):
    y = y.dropna()
    if y.dtype.kind in "fi" and y.nunique() > 20:
        return "regression"
    return "classification"


def build_pre(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, sparse_output=False))]), cat))
    return ColumnTransformer(parts)


def make_model(kind, clf, seed):
    if kind.startswith("Simple"):
        return LogisticRegression(max_iter=1000, random_state=seed) if clf else Ridge()
    return (RandomForestClassifier if clf else RandomForestRegressor)(n_estimators=150, random_state=seed, n_jobs=-1)


# ----------------------------------------------------------------------------- analysis
@st.cache_data(show_spinner=False)
def data_quality(df, target):
    X = df.drop(columns=[target]); num = X.select_dtypes("number")
    out = {}
    for c in num.columns:
        s = num[c].dropna()
        if len(s) > 3:
            q1, q3 = s.quantile([.25, .75]); i = q3 - q1
            out[c] = float(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).mean() * 100)
    stds = num.std().replace(0, np.nan).dropna()
    return dict(rows=len(df), cols=X.shape[1], miss=(df.isna().mean() * 100).round(2), dup=int(df.duplicated().sum()),
                const=[c for c in X if X[c].nunique(dropna=False) <= 1],
                idlike=[c for c in X if len(X) > 20 and X[c].nunique() == len(X) and X[c].dtype.kind in "iuOS"],
                highcard=[c for c in X.select_dtypes(exclude="number") if X[c].nunique() > 50],
                cat=[c for c in X if c not in num.columns], outliers=out,
                scale_ratio=float(stds.max() / stds.min()) if len(stds) > 1 else 1.0,
                skewed=[c for c in num if abs(num[c].skew()) > 2], target_missing=int(df[target].isna().sum()))


@st.cache_data(show_spinner=False)
def leakage_scan(df, target, task):
    d = df.dropna(subset=[target])
    d = d.sample(4000, random_state=0) if len(d) > 4000 else d
    clf = task == "classification"
    y = d[target].astype(str).astype("category").cat.codes if clf else d[target].astype(float)
    k = max(y.nunique(), 2); rows = []
    for c in [c for c in d.columns if c != target][:80]:
        s = d[c]
        v = s.fillna(s.median()) if s.dtype.kind in "iuf" else s.astype(str).astype("category").cat.codes
        if v.nunique() < 2:
            continue
        m = DecisionTreeClassifier(max_depth=5, random_state=0) if clf else DecisionTreeRegressor(max_depth=5, random_state=0)
        sc = cross_val_score(m, v.values.reshape(-1, 1), y, cv=KFold(3, shuffle=True, random_state=0),
                             scoring="balanced_accuracy" if clf else "r2").mean()
        strength = max(0.0, (sc - 1 / k) / (1 - 1 / k)) if clf else max(0.0, sc)
        same = bool(s.astype(str).reset_index(drop=True).equals(d[target].astype(str).reset_index(drop=True)))
        rows.append(dict(col=c, strength=float(strength), hint=any(w in c.lower() for w in LEAK_WORDS), identical=same))
    return sorted(rows, key=lambda r: -r["strength"])


def sc_fn(clf):
    return balanced_accuracy_score if clf else r2_score


@st.cache_data(show_spinner=False)
def evaluate(df, target, task, kind, seed, test_size):
    clf = task == "classification"
    d = df[df[target].notna()].reset_index(drop=True)
    d = d.sample(6000, random_state=seed).reset_index(drop=True) if len(d) > 6000 else d
    X, y = d.drop(columns=[target]), d[target]
    y = y.astype(str) if clf else y.astype(float)
    ok_strat = clf and y.value_counts().min() >= 5
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if ok_strat else None)
    mk = lambda: Pipeline([("pre", build_pre(X)), ("m", make_model(kind, clf, seed))])
    pipe = mk().fit(Xtr, ytr); ptr, pva = pipe.predict(Xtr), pipe.predict(Xva)
    S = sc_fn(clf); R = dict(clf=clf, n_train=len(Xtr), n_val=len(Xva), classes=sorted(y.unique().tolist()) if clf else [])
    R["score_name"] = "Balanced accuracy" if clf else "R²"
    R["score_tr"], R["score_va"] = float(S(ytr, ptr)), float(S(yva, pva))
    R["baseline"] = 1 / max(y.nunique(), 2) if clf else 0.0
    if clf:
        m = dict(Accuracy=accuracy_score(yva, pva), **{"Balanced accuracy": R["score_va"]},
                 **{"Precision (macro)": precision_score(yva, pva, average="macro", zero_division=0),
                    "Recall (macro)": recall_score(yva, pva, average="macro", zero_division=0),
                    "F1 (macro)": f1_score(yva, pva, average="macro", zero_division=0)})
        try:
            pr = pipe.predict_proba(Xva)
            m["ROC AUC"] = roc_auc_score(yva, pr[:, 1]) if pr.shape[1] == 2 else roc_auc_score(yva, pr, multi_class="ovr", labels=pipe.classes_)
        except Exception:
            pass
        R["metrics"] = m; labels = sorted(y.unique())
        R["cm"] = confusion_matrix(yva, pva, labels=labels).tolist(); R["labels"] = labels
        rep = classification_report(yva, pva, output_dict=True, zero_division=0)
        R["per_class"] = pd.DataFrame({k: v for k, v in rep.items() if k in labels}).T[["precision", "recall", "f1-score", "support"]]
        R["majority_acc"] = float(y.value_counts(normalize=True).iloc[0])
    else:
        R["metrics"] = dict(**{"R²": R["score_va"]}, RMSE=float(np.sqrt(mean_squared_error(yva, pva))), MAE=float(mean_absolute_error(yva, pva)))
        R["resid"] = (yva.values - pva).tolist(); R["pred"] = pva.tolist(); R["true"] = yva.values.tolist()
    # preprocessing fitted on ALL rows (incorrect) vs train only (correct)
    pa = build_pre(X).fit(X); mm = make_model(kind, clf, seed).fit(pa.transform(Xtr), ytr)
    R["score_va_leakyprep"] = float(S(yva, mm.predict(pa.transform(Xva))))
    hs = pd.util.hash_pandas_object(Xtr, index=False)
    R["row_overlap"] = float(pd.util.hash_pandas_object(Xva, index=False).isin(set(hs)).mean())
    cv = (StratifiedKFold if ok_strat else KFold)(4, shuffle=True, random_state=seed)
    sizes, trs, vas = learning_curve(mk(), X, y, cv=cv, train_sizes=np.linspace(.15, 1, 6), scoring="balanced_accuracy" if clf else "r2", n_jobs=1)
    R["lc"] = dict(sizes=sizes.tolist(), tr=trs.tolist(), va=vas.tolist())
    R["cv_mean"], R["cv_std"] = float(vas[-1].mean()), float(vas[-1].std())
    sub = Xva.iloc[:800]; ysub = yva.iloc[:800]
    pi = permutation_importance(pipe, sub, ysub, n_repeats=3, random_state=seed, scoring="balanced_accuracy" if clf else "r2")
    R["imp"] = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(15).to_dict()
    return R


@st.cache_data(show_spinner=False)
def train_torch(df, target, task, seed, test_size, epochs, lr, hidden, wd):
    torch.manual_seed(seed); np.random.seed(seed)
    clf = task == "classification"
    d = df[df[target].notna()].reset_index(drop=True)
    d = d.sample(6000, random_state=seed) if len(d) > 6000 else d
    X, y = d.drop(columns=[target]), d[target]
    y = y.astype(str) if clf else y.astype(float)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if clf and y.value_counts().min() >= 5 else None)
    pre = build_pre(X).fit(Xtr)
    A, B = torch.tensor(pre.transform(Xtr), dtype=torch.float32), torch.tensor(pre.transform(Xva), dtype=torch.float32)
    if clf:
        cl = sorted(ytr.unique()); mp = {c: i for i, c in enumerate(cl)}
        yt = torch.tensor(ytr.map(mp).values); keep = yva.map(mp).notna().values
        B, yv = B[torch.tensor(keep)], torch.tensor(yva[keep].map(mp).values.astype(int)); out, lf = len(cl), nn.CrossEntropyLoss()
    else:
        mu, sd = ytr.mean(), ytr.std() or 1.0
        yt = torch.tensor(((ytr - mu) / sd).values, dtype=torch.float32); yv = torch.tensor(((yva - mu) / sd).values, dtype=torch.float32); out, lf = 1, nn.MSELoss()
    net = nn.Sequential(nn.Linear(A.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd); rows = []
    def ev(Z, t):
        with torch.no_grad():
            o = net(Z); o = o if clf else o.squeeze(1)
            l = lf(o, t).item()
            s = (o.argmax(1) == t).float().mean().item() if clf else 1 - ((o - t) ** 2).sum().item() / ((t - t.mean()) ** 2).sum().item()
        return l, s
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(len(A))
        for i in range(0, len(A), 64):
            idx = perm[i:i + 64]; opt.zero_grad(); o = net(A[idx]); o = o if clf else o.squeeze(1)
            lf(o, yt[idx]).backward(); opt.step()
        net.eval(); (tl, ts), (vl, vs) = ev(A, yt), ev(B, yv)
        rows.append(dict(epoch=ep, train_loss=tl, val_loss=vl, train_score=ts, val_score=vs))
    return pd.DataFrame(rows)


def diagnose_curve(tr, va, kind, source):
    """kind: 'loss' or 'score' (accuracy-like, 0-1). Everything is derived from the numbers supplied."""
    tr, va = np.asarray(tr, float), np.asarray(va, float); n = len(tr)
    if kind == "score":
        tr, va = 1 - tr, 1 - va
    unit = "loss" if kind == "loss" else "error (1 − score)"
    k = max(3, n // 4); slope = lambda a: float(np.polyfit(np.arange(len(a)), a, 1)[0])
    best = int(np.argmin(va)); out = []; ts, vs = slope(tr[-k:]), slope(va[-k:])
    ev = f"{source}: train {unit} {tr[0]:.4f} → {tr[-1]:.4f}; validation {va[0]:.4f} → {va[-1]:.4f}; best validation at epoch {best + 1}/{n}."
    if not np.isfinite(tr).all() or not np.isfinite(va).all() or tr[-1] > tr[0] * 1.5:
        out.append(F("Critical", "Training diverged or produced invalid values", ev,
                     "The model is not learning: loss grows or becomes NaN/inf.", ["Learning rate too high", "Unscaled inputs or exploding gradients", "Bad labels or loss function mismatch"],
                     ["Lower the learning rate 10×", "Standardise inputs", "Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)"]))
        return out
    if best < 0.8 * (n - 1) and va[-1] > va[best] * 1.05 and vs > 0 and ts <= 0.001:
        out.append(F("Critical" if va[-1] > va[best] * 1.25 else "Warning", "Overfitting: validation got worse while training kept improving",
                     ev + f" Validation {unit} rose {100 * (va[-1] / va[best] - 1):.1f}% above its minimum.",
                     "The model started memorising training rows instead of learning general patterns.",
                     ["Model too large for the data", "Trained too long without early stopping", "No regularisation", "Too few training rows"],
                     ["Stop at the epoch with best validation loss (early stopping)", "Add weight decay / dropout", "Shrink the network or add data / augmentation"],
                     "best, wait = float('inf'), 0\nfor epoch in range(max_epochs):\n    train_one_epoch(); val_loss = evaluate()\n    if val_loss < best - 1e-4:\n        best, wait = val_loss, 0; torch.save(model.state_dict(), 'best.pt')\n    else:\n        wait += 1\n        if wait >= 5: break  # early stopping\nmodel.load_state_dict(torch.load('best.pt'))"))
    elif tr[-1] > 0.7 * tr[0] and abs(va[-1] - tr[-1]) < 0.25 * max(tr[-1], 1e-9):
        out.append(F("Warning", "Underfitting: both curves stay high", ev + f" Train {unit} only improved {100 * (1 - tr[-1] / tr[0]):.1f}%.",
                     "The model is too simple or hasn't trained enough to capture the signal.",
                     ["Model capacity too low", "Too few epochs / learning rate too small", "Features lack signal", "Over-regularised"],
                     ["Increase hidden units/layers or use a more flexible model", "Train longer or raise the learning rate", "Engineer more informative features", "Reduce regularisation"]))
    elif va[-1] - tr[-1] > 0.5 * max(tr[-1], 1e-9) and va[-1] - tr[-1] > 0.05:
        out.append(F("Warning", "Large generalisation gap", ev + f" Gap = {va[-1] - tr[-1]:.4f} ({100 * (va[-1] - tr[-1]) / max(tr[-1], 1e-9):.0f}% of train {unit}).",
                     "The model performs much better on data it has seen than on new data.", ["Too little data", "High-variance model", "Train/validation distribution mismatch"],
                     ["Add regularisation or early stopping", "Collect more data", "Check that train and validation come from the same distribution"]))
    d = np.abs(np.diff(va[-k:]))
    if len(d) and d.mean() > 0.15 * max(abs(va[-k:]).mean(), 1e-9):
        out.append(F("Info", "Noisy validation curve", f"Mean epoch-to-epoch change in validation {unit} over the last {k} epochs is {d.mean():.4f} ({100 * d.mean() / max(abs(va[-k:]).mean(), 1e-9):.0f}% of its level).",
                     "Validation results swing a lot between epochs.", ["Small validation set", "Learning rate too high", "Small batch size"],
                     ["Use a larger validation set or cross-validation", "Lower the learning rate or add a scheduler"]))
    if not out:
        out.append(F("OK", "Learning curves look healthy", ev, "Both curves improve together with a small gap.", [], []))
    return out


def build_findings(df, target, task, dq, leak, R, T, kind):
    clf = task == "classification"; out = []; drop = []; ctx = dict(drop=[], balance=False, regularise=False)
    sus = [r for r in leak if r["identical"] or r["strength"] >= .8 or (r["hint"] and r["strength"] >= .5)]
    if sus:
        drop += [r["col"] for r in sus]
        ev = "; ".join(f"`{r['col']}` alone predicts the target with strength {r['strength']:.2f}" + (" (identical to target)" if r["identical"] else "") for r in sus[:5])
        out.append(F("Critical" if any(r["identical"] or r["strength"] >= .9 for r in sus) else "Warning", "Possible data leakage", ev + f". Validation score is {R['score_va']:.3f}.",
                     "A column that contains the answer (or is only known after the outcome) makes the model look great in testing and fail in production.",
                     ["Feature created after the outcome happened", "Target copied/transformed into a feature", "Aggregates computed over the whole dataset"],
                     ["Ask when each flagged column becomes available in real use", "Drop columns that are unavailable at prediction time", "Re-run diagnosis to see the honest score"],
                     f"leaky = {[r['col'] for r in sus]}\nX = df.drop(columns=['{target}'] + leaky)"))
    if dq["idlike"]:
        drop += dq["idlike"]
        out.append(F("Warning", "ID-like columns present", f"Columns with a unique value per row: {dq['idlike']}.", "IDs carry no general signal and let models memorise rows.", ["Row identifiers left in features"], ["Drop them before training"], f"X = X.drop(columns={dq['idlike']})"))
    if dq["const"]:
        drop += dq["const"]
        out.append(F("Info", "Constant columns", f"{dq['const']} have a single value.", "They add no information.", ["Exported columns never populated"], ["Drop them"], f"X = X.drop(columns={dq['const']})"))
    m = dq["miss"]; mm = m[m > 0]
    if len(mm):
        worst = mm.sort_values(ascending=False)
        out.append(F("Warning" if worst.iloc[0] > 20 else "Info", "Missing values", f"{len(mm)} column(s) have gaps; worst: " + ", ".join(f"`{c}` {v:.1f}%" for c, v in worst.head(4).items()) + ".",
                     "Most models cannot handle NaN, and naive fills can distort results.", ["Optional fields", "Join mismatches", "Sensor/ETL failures"],
                     ["Drop columns that are mostly empty (>50%)", "Impute inside the training pipeline only (never on the full dataset)", "Consider a missing-indicator feature"],
                     "from sklearn.impute import SimpleImputer\nSimpleImputer(strategy='median', add_indicator=True)"))
    if dq["target_missing"]:
        out.append(F("Warning", "Rows with missing target", f"{dq['target_missing']} rows have no target and were excluded.", "Unlabelled rows can't train or score a supervised model.", ["Incomplete labelling"], ["Label them or drop them explicitly"], f"df = df.dropna(subset=['{target}'])"))
    if dq["dup"] or R["row_overlap"] > 0:
        out.append(F("Warning" if dq["dup"] / dq["rows"] > .01 or R["row_overlap"] > .02 else "Info", "Duplicate rows / train-validation overlap",
                     f"{dq['dup']} duplicate rows ({100 * dq['dup'] / dq['rows']:.1f}%); {100 * R['row_overlap']:.1f}% of validation rows also appear in training.",
                     "Identical rows on both sides of the split inflate validation scores.", ["Repeated records", "Splitting after oversampling/augmentation"],
                     ["Deduplicate before splitting", "Oversample only the training fold"], "df = df.drop_duplicates()\nX_train, X_test, y_train, y_test = train_test_split(df.drop(columns='%s'), df['%s'], test_size=0.2, random_state=42)" % (target, target)))
    if clf:
        vc = df[target].value_counts(normalize=True); mn = vc.iloc[-1]; pc = R["per_class"]; low = pc["recall"].idxmin()
        if mn < .2 or vc.iloc[0] > .8:
            ctx["balance"] = True
            out.append(F("Critical" if mn < .1 else "Warning", "Class imbalance", f"Class `{vc.index[-1]}` is {100 * mn:.1f}% of rows; majority class is {100 * vc.iloc[0]:.1f}%. Predicting the majority gives {100 * R['majority_acc']:.1f}% accuracy; recall for `{low}` is {pc.loc[low, 'recall']:.2f}.",
                         "Accuracy looks high but the rare class is often missed.", ["Rare event by nature", "Biased sampling"],
                         ["Use stratified splits", "Judge by recall/F1/PR-AUC, not accuracy", "Use class_weight='balanced' or resample the training fold only"],
                         "from sklearn.model_selection import train_test_split\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)\nmodel = LogisticRegression(class_weight='balanced', max_iter=1000)"))
    pr = []
    if dq["cat"]:
        pr.append(f"non-numeric columns {dq['cat'][:5]} need encoding")
    if dq["scale_ratio"] > 100:
        pr.append(f"feature std ratio is {dq['scale_ratio']:.0f}× (scale-sensitive models need scaling)")
    if dq["skewed"]:
        pr.append(f"highly skewed columns {dq['skewed'][:5]}")
    if dq["highcard"]:
        pr.append(f"high-cardinality columns {dq['highcard']}")
    delta = R["score_va_leakyprep"] - R["score_va"]
    if delta > .02:
        pr.append(f"fitting preprocessing on all rows changed validation {R['score_name']} by {delta:+.3f}")
    if pr:
        out.append(F("Warning" if delta > .02 else "Info", "Preprocessing issues", "; ".join(pr) + ".", "Wrong or leaky preprocessing quietly biases results.",
                     ["Scaler/imputer fit on the full dataset before splitting", "Categoricals left as text"], ["Put every transform inside a sklearn Pipeline so it is fit on training data only", "Log-transform skewed columns if useful"],
                     "pipe = Pipeline([('pre', ColumnTransformer([...])), ('model', model)])\npipe.fit(X_train, y_train)  # preprocessing learns from train only"))
    gap, base = R["score_tr"] - R["score_va"], R["baseline"]
    if gap > .1:
        ctx["regularise"] = True
        out.append(F("Critical" if gap > .2 else "Warning", "Overfitting", f"{kind}: train {R['score_name']} {R['score_tr']:.3f} vs validation {R['score_va']:.3f} (gap {gap:.3f}); {R['n_train']} training rows, {df.shape[1] - 1} features.",
                     "The model memorises the training set instead of generalising.", ["Too many features for the number of rows", "Model too flexible", "Noise fitted"],
                     ["Use stronger regularisation / shallower trees", "Reduce features (selection)", "Collect more data", "Use cross-validation to tune"],
                     "RandomForestClassifier(max_depth=6, min_samples_leaf=5, random_state=42)\n# or LogisticRegression(C=0.1)"))
    elif R["score_tr"] - base < .15 and not sus:
        out.append(F("Warning", "Underfitting", f"{kind}: train {R['score_name']} is only {R['score_tr']:.3f} (baseline {base:.3f}), with train–validation gap {gap:.3f}.",
                     "Even the training data isn't fit well: the model can't capture the pattern.", ["Linear model on non-linear data", "Missing informative features", "Over-regularisation"],
                     ["Try a more flexible model (random forest / gradient boosting / MLP)", "Add interaction/polynomial features", "Reduce regularisation"],
                     "from sklearn.ensemble import HistGradientBoostingRegressor\nmodel = HistGradientBoostingRegressor()" if not clf else "from sklearn.ensemble import HistGradientBoostingClassifier\nmodel = HistGradientBoostingClassifier()"))
    if R["score_va"] - base < .05:
        out.append(F("Critical", "Poor validation performance: no signal beyond baseline", f"Validation {R['score_name']} {R['score_va']:.3f} vs baseline {base:.3f}.", "The features currently don't predict the target better than guessing.",
                     ["Weak/irrelevant features", "Target noise", "Label–feature misalignment"], ["Verify labels", "Engineer domain features", "Check for joins that broke alignment"]))
    elif R["cv_std"] > .05:
        out.append(F("Warning", "Unstable validation score", f"4-fold CV {R['score_name']}: {R['cv_mean']:.3f} ± {R['cv_std']:.3f}.", "Results vary a lot between folds, so one split can mislead.",
                     ["Small dataset", "Rare classes"], ["Use repeated stratified k-fold", "Report the mean and spread"], "cross_val_score(pipe, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42))"))
    lc = R["lc"]; va_m = np.mean(lc["va"], 1)
    if va_m[-1] - va_m[-2] > .01 and gap <= .1:
        out.append(F("Info", "More data would likely help", f"Validation {R['score_name']} still rising at the largest size ({va_m[-2]:.3f} → {va_m[-1]:.3f}).", "The learning curve hasn't flattened.", [], ["Collect more rows before tuning the model heavily"]))
    if T is not None and len(T) >= 5:
        out += [dict(f, title="PyTorch MLP · " + f["title"]) for f in diagnose_curve(T.train_loss, T.val_loss, "loss", "PyTorch MLP") if f["sev"] != "OK"]
    if not out:
        out.append(F("OK", "No major problems detected", f"Validation {R['score_name']} {R['score_va']:.3f}, gap {gap:.3f}, CV {R['cv_mean']:.3f} ± {R['cv_std']:.3f}.", "All checks passed.", [], []))
    ctx["drop"] = sorted(set(drop))
    return sorted(out, key=lambda f: ORDER[f["sev"]]), ctx


def corrected_code(target, task, ctx, kind):
    clf = task == "classification"
    model = ("LogisticRegression(max_iter=1000" + (", class_weight='balanced'" if ctx["balance"] else "") + ")") if clf else "HistGradientBoostingRegressor()"
    if clf and kind.startswith("Random"):
        model = "RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5" + (", class_weight='balanced'" if ctx["balance"] else "") + ", random_state=42)"
    imps = "from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor\nfrom sklearn.linear_model import LogisticRegression"
    return f'''import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import {"classification_report" if clf else "r2_score, mean_squared_error"}
from sklearn.model_selection import train_test_split, {"StratifiedKFold, " if clf else "KFold, "}cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
{imps}

df = pd.read_csv("your_data.csv").drop_duplicates().dropna(subset=["{target}"])
leaky_or_id = {ctx["drop"]}            # columns flagged by the diagnosis
X, y = df.drop(columns=["{target}"] + leaky_or_id), df["{target}"]

num = X.select_dtypes("number").columns
cat = X.columns.difference(num)
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
])
pipe = Pipeline([("pre", pre), ("model", {model})])  # preprocessing is fit on training data only

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42{", stratify=y" if clf else ""})
pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)
{"print(classification_report(y_te, pred))" if clf else "print('R2', r2_score(y_te, pred), 'RMSE', mean_squared_error(y_te, pred) ** 0.5)"}
cv = {"StratifiedKFold(5, shuffle=True, random_state=42)" if clf else "KFold(5, shuffle=True, random_state=42)"}
print(cross_val_score(pipe, X, y, cv=cv, scoring="{"balanced_accuracy" if clf else "r2"}").round(3))
'''


def render_findings(fs):
    for f in fs:
        c = COLORS[f["sev"]]
        st.markdown(f"<div class='finding' style='--c:{c}'><b>{ICON[f['sev']]} {f['sev']} · {f['title']}</b><br>{f['evidence']}</div>", unsafe_allow_html=True)
        with st.expander("Why it matters, root causes and fix"):
            st.write(f["why"])
            if f["causes"]:
                st.markdown("**Possible root causes**\n" + "\n".join(f"- {c}" for c in f["causes"]))
            if f["steps"]:
                st.markdown("**Step-by-step fix**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(f["steps"], 1)))
            if f["code"]:
                st.code(f["code"], language="python")


def report_md(title, fs, facts, code):
    L = [f"# ML Pipeline Doctor AI report: {title}", ""]
    L += ["## Measured facts"] + [f"- **{k}**: {v}" for k, v in facts.items()] + ["", "## Findings"]
    for f in fs:
        L += [f"### {ICON[f['sev']]} [{f['sev']}] {f['title']}", f["evidence"], "", f["why"]]
        L += (["", "Root causes:"] + [f"- {c}" for c in f["causes"]]) if f["causes"] else []
        L += (["", "Steps:"] + [f"{i}. {s}" for i, s in enumerate(f["steps"], 1)]) if f["steps"] else []
        L += (["", "```python", f["code"], "```"]) if f["code"] else []
        L.append("")
    if code:
        L += ["## Corrected pipeline", "```python", code, "```"]
    return "\n".join(L)


def explain(fs, facts, question, key, model):
    payload = json.dumps(dict(facts=facts, findings=[{k: f[k] for k in ("sev", "title", "evidence", "causes", "steps")} for f in fs]), default=str)[:12000]
    if key:
        try:
            import anthropic
            r = anthropic.Anthropic(api_key=key).messages.create(model=model, max_tokens=900, system="You explain ML diagnostics to beginners. Use ONLY the numbers in the JSON. Never invent metrics. If unsure, say so.",
                                                                 messages=[{"role": "user", "content": f"Diagnostics JSON:\n{payload}\n\nQuestion: {question or 'Explain these results simply and give a prioritised plan.'}"}])
            return "".join(b.text for b in r.content if b.type == "text"), "LLM (grounded in your computed diagnostics)"
        except Exception as e:
            st.warning(f"LLM call failed ({type(e).__name__}); using the built-in explainer instead.")
    words = set(re.findall(r"\w+", (question or "").lower()))
    rel = [f for f in fs if words & set(re.findall(r"\w+", (f["title"] + f["why"]).lower()))] if question else fs
    rel = rel or fs
    txt = ["**Plain-English summary** (rule-based, computed from your results)\n"]
    for i, f in enumerate(rel[:6], 1):
        txt.append(f"**{i}. {f['title']}** ({f['sev']}): {f['why']} Evidence: {f['evidence']}" + (f" First step: {f['steps'][0]}." if f["steps"] else ""))
    return "\n\n".join(txt), "Built-in explainer"


def fig_style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2)); return fig


def curve_fig(x, tr, va, title, xt, yt):
    fig = go.Figure()
    for name, v, col in (("Training", tr, "#14675f"), ("Validation", va, "#e08a1e")):
        v = np.asarray(v)
        if v.ndim == 2:
            m, s = v.mean(1), v.std(1)
            fig.add_trace(go.Scatter(x=x, y=m + s, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=m - s, fill="tonexty", line=dict(width=0), showlegend=False, hoverinfo="skip", fillcolor="rgba(127,127,127,.18)"))
            v = m
        fig.add_trace(go.Scatter(x=x, y=v, name=name, mode="lines+markers", line=dict(color=col, width=3)))
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt); return fig_style(fig)


# ----------------------------------------------------------------------------- UI
st.markdown("<div class='hero'><h1>🩺 ML Pipeline Doctor AI</h1><p>Upload a dataset or paste training metrics. Every finding below is calculated from your data.</p></div>", unsafe_allow_html=True)
with st.sidebar:
    st.header("1 · Choose input")
    mode = st.radio("Input", ["Sample dataset", "Upload CSV", "Manual training metrics"], label_visibility="collapsed")
    st.header("AI explanation (optional)")
    api_key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""), help="Leave empty to use the built-in explainer.")
    llm_model = st.text_input("Model", value="claude-sonnet-5-5")

# ---- manual metrics
if mode == "Manual training metrics":
    st.subheader("Enter training metrics")
    kind_m = st.radio("Metric type", ["loss", "score"], horizontal=True, format_func=lambda k: "Loss (lower is better)" if k == "loss" else "Accuracy-like score 0–1 (higher is better)")
    sample = {"loss": ("1.0,0.7,0.5,0.38,0.3,0.24,0.19,0.15,0.12,0.1", "1.05,0.8,0.62,0.55,0.53,0.54,0.58,0.63,0.7,0.78"),
              "score": ("0.55,0.7,0.8,0.86,0.9,0.93,0.95,0.97", "0.54,0.68,0.75,0.78,0.79,0.78,0.77,0.75")}[kind_m]
    c1, c2 = st.columns(2)
    t_txt = c1.text_area("Training values, one per epoch (comma or newline separated)", sample[0], height=120, key=f"t{kind_m}")
    v_txt = c2.text_area("Validation values", sample[1], height=120, key=f"v{kind_m}")
    try:
        parse = lambda s: [float(x) for x in re.split(r"[,\s;]+", s.strip()) if x]
        tr, va = parse(t_txt), parse(v_txt)
        if len(tr) != len(va): raise ValueError(f"Training has {len(tr)} values but validation has {len(va)}. They must match.")
        if len(tr) < 5: raise ValueError("Enter at least 5 epochs.")
        if kind_m == "score" and not (0 <= min(tr + va) and max(tr + va) <= 1): raise ValueError("Scores must be between 0 and 1.")
    except ValueError as e:
        st.error(f"Couldn't read the metrics: {e}"); st.stop()
    fs = diagnose_curve(tr, va, kind_m, "Your metrics")
    ep = list(range(1, len(tr) + 1))
    st.plotly_chart(curve_fig(ep, tr, va, "Training vs validation", "Epoch", kind_m.capitalize()), use_container_width=True)
    render_findings(fs)
    facts = {"epochs": len(tr), "final train": tr[-1], "final validation": va[-1], "best validation epoch": int(np.argmin(va) + 1) if kind_m == "loss" else int(np.argmax(va) + 1)}
    t1, t2 = st.tabs(["AI explanation", "Report"])
    with t1:
        q = st.text_input("Ask a question (optional)", key="qm")
        if st.button("Explain", key="bm"):
            a, src = explain(fs, facts, q, api_key, llm_model); st.caption(src); st.markdown(a)
    with t2:
        md = report_md("manual metrics", fs, facts, "")
        st.download_button("Download report (.md)", md, "diagnostic_report.md"); st.download_button("Download JSON", json.dumps(dict(facts=facts, findings=fs), default=str), "diagnostic_report.json")
    st.stop()

# ---- dataset modes
df = None
try:
    if mode == "Sample dataset":
        name = st.sidebar.selectbox("Sample", ["Healthy classification", "Data leakage", "Imbalance + missing values", "Overfitting (few rows, many features)", "Underfitting (non-linear regression)"])
        df = make_sample(name); st.sidebar.caption("Synthetic data generated locally with scikit-learn.")
    else:
        up = st.sidebar.file_uploader("CSV file", type=["csv"])
        if up is None:
            st.info("Upload a CSV in the sidebar, or pick **Sample dataset** to see the doctor at work."); st.stop()
        if up.size > 50 * 1024 ** 2: raise ValueError("File is larger than 50 MB. Sample it first.")
        for enc in ("utf-8", "latin-1"):
            try: up.seek(0); df = pd.read_csv(up, encoding=enc); break
            except UnicodeDecodeError: continue
        if df is None or df.empty: raise ValueError("The CSV is empty or unreadable.")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = df.columns.astype(str)
    if len(df) > 20000: df = df.sample(20000, random_state=0).reset_index(drop=True); st.sidebar.warning("Large file: analysing a 20,000-row sample.")
    if len(df) < 30 or df.shape[1] < 2: raise ValueError("Need at least 30 rows and 2 columns.")
except Exception as e:
    st.error(f"Couldn't load the data: {e}"); st.stop()

st.sidebar.header("2 · Configure")
target = st.sidebar.selectbox("Target column", df.columns, index=len(df.columns) - 1)
task = st.sidebar.selectbox("Task", ["auto", "classification", "regression"])
task = infer_task(df[target]) if task == "auto" else task
if task == "regression" and df[target].dtype.kind not in "iuf":
    st.error("Regression needs a numeric target. Choose classification or a different column."); st.stop()
if task == "classification" and df[target].nunique() > 50:
    st.error(f"Target has {df[target].nunique()} distinct values; it looks continuous. Switch the task to regression."); st.stop()
if task == "classification" and df[target].nunique() < 2:
    st.error("Target has a single class."); st.stop()
kind = st.sidebar.selectbox("Model to diagnose", ["Simple linear model", "Random forest"])
test_size = st.sidebar.slider("Validation share", .1, .4, .2, .05)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
use_t = st.sidebar.checkbox("Train PyTorch MLP for epoch curves", value=HAS_TORCH, disabled=not HAS_TORCH)
ep_n = st.sidebar.slider("Epochs", 10, 100, 40, disabled=not use_t)
lr = st.sidebar.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 1e-3, disabled=not use_t)
hid = st.sidebar.select_slider("Hidden units", [8, 16, 32, 64, 128, 256], 64, disabled=not use_t)
wd = st.sidebar.select_slider("Weight decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2], 0.0, disabled=not use_t)
if not HAS_TORCH: st.sidebar.caption("PyTorch not installed: epoch curves disabled. Run `pip install torch`.")

try:
    with st.spinner("Running diagnostics…"):
        dq = data_quality(df, target); leak = leakage_scan(df, target, task)
        R = evaluate(df, target, task, kind, int(seed), test_size)
        T = train_torch(df, target, task, int(seed), test_size, ep_n, lr, hid, wd) if use_t else None
        fs, ctx = build_findings(df, target, task, dq, leak, R, T, kind)
except Exception as e:
    st.error(f"Diagnosis failed: {type(e).__name__}: {e}. Check the target column, data types and that each class has at least 5 rows.")
    with st.expander("Technical details"): st.code(traceback.format_exc())
    st.stop()

facts = {"rows": dq["rows"], "features": dq["cols"], "task": task, "model": kind, f"train {R['score_name']}": round(R["score_tr"], 4),
         f"validation {R['score_name']}": round(R["score_va"], 4), "baseline": round(R["baseline"], 4), "CV mean": round(R["cv_mean"], 4), "CV std": round(R["cv_std"], 4),
         "duplicates": dq["dup"], "missing cells %": round(float(df.isna().mean().mean() * 100), 2), **{k: round(v, 4) for k, v in R["metrics"].items()}}
code = corrected_code(target, task, ctx, kind)
n_bad = sum(f["sev"] in ("Critical", "Warning") for f in fs)
c = st.columns(4)
c[0].metric("Issues found", n_bad, help="Critical + warning findings")
c[1].metric(f"Validation {R['score_name']}", f"{R['score_va']:.3f}", f"{R['score_va'] - R['baseline']:+.3f} vs baseline")
c[2].metric("Train–val gap", f"{R['score_tr'] - R['score_va']:.3f}")
c[3].metric("CV mean ± std", f"{R['cv_mean']:.3f} ± {R['cv_std']:.3f}")

tabs = st.tabs(["🩺 Diagnosis", "🧪 Data quality", "📈 Learning curves", "🎯 Metrics", "🤖 AI explain", "📄 Report"])
with tabs[0]:
    render_findings(fs)
    st.subheader("Corrected pipeline"); st.code(code, language="python")
with tabs[1]:
    a, b = st.columns(2)
    m = dq["miss"][dq["miss"] > 0].sort_values()
    if len(m): a.plotly_chart(fig_style(px.bar(m, orientation="h", title="Missing values (%)", labels={"value": "%", "index": ""})), use_container_width=True)
    else: a.success("No missing values.")
    if R["clf"]:
        vc = df[target].astype(str).value_counts(); b.plotly_chart(fig_style(px.bar(x=vc.index, y=vc.values, title="Class distribution", labels={"x": "Class", "y": "Rows"})), use_container_width=True)
    else: b.plotly_chart(fig_style(px.histogram(df, x=target, title="Target distribution")), use_container_width=True)
    lk = pd.DataFrame(leak).head(12)
    if len(lk): st.plotly_chart(fig_style(px.bar(lk.iloc[::-1], x="strength", y="col", orientation="h", title="Leakage scan: single-feature predictive strength (≥0.8 is suspicious)", range_x=[0, 1])), use_container_width=True)
    st.dataframe(pd.DataFrame({"Check": ["Rows", "Features", "Duplicate rows", "Constant columns", "ID-like columns", "High-cardinality columns", "Feature scale ratio (max/min std)", "Highly skewed columns"],
                               "Result": [dq["rows"], dq["cols"], dq["dup"], str(dq["const"]), str(dq["idlike"]), str(dq["highcard"]), f"{dq['scale_ratio']:.1f}", str(dq["skewed"])]}), hide_index=True, use_container_width=True)
    if dq["outliers"]:
        o = pd.Series(dq["outliers"]).sort_values(ascending=False).head(10); st.plotly_chart(fig_style(px.bar(o, title="Outliers by column (IQR rule, % of values)", labels={"value": "%", "index": ""})), use_container_width=True)
with tabs[2]:
    lc = R["lc"]; st.plotly_chart(curve_fig(lc["sizes"], lc["tr"], lc["va"], f"Learning curve: {R['score_name']} vs training size (±1 std, 4-fold CV)", "Training rows", R["score_name"]), use_container_width=True)
    if T is not None:
        st.plotly_chart(curve_fig(T.epoch, T.train_loss, T.val_loss, "PyTorch MLP: loss per epoch", "Epoch", "Loss"), use_container_width=True)
        st.plotly_chart(curve_fig(T.epoch, T.train_score, T.val_score, "PyTorch MLP: " + ("accuracy" if R["clf"] else "R²") + " per epoch", "Epoch", "Score"), use_container_width=True)
with tabs[3]:
    st.dataframe(pd.DataFrame({"Metric": R["metrics"].keys(), "Validation": [round(v, 4) for v in R["metrics"].values()]}), hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    if R["clf"]:
        a.plotly_chart(fig_style(px.imshow(R["cm"], x=R["labels"], y=R["labels"], text_auto=True, color_continuous_scale="Teal", title="Confusion matrix", labels=dict(x="Predicted", y="Actual"))), use_container_width=True)
        b.dataframe(R["per_class"].round(3), use_container_width=True)
    else:
        a.plotly_chart(fig_style(px.scatter(x=R["true"], y=R["pred"], title="Predicted vs actual", labels={"x": "Actual", "y": "Predicted"})), use_container_width=True)
        b.plotly_chart(fig_style(px.histogram(x=R["resid"], title="Residuals")), use_container_width=True)
    imp = pd.Series(R["imp"]).iloc[::-1]; st.plotly_chart(fig_style(px.bar(imp, orientation="h", title="Permutation importance on validation data (a dominant single feature can hint at leakage)")), use_container_width=True)
with tabs[4]:
    q = st.text_input("Ask about your results (optional)", placeholder="e.g. Why is my model overfitting?")
    if st.button("Explain my results", type="primary"):
        with st.spinner("Thinking…"):
            a, src = explain(fs, facts, q, api_key, llm_model)
        st.caption(src); st.markdown(a)
with tabs[5]:
    md = report_md(f"{task} on `{target}`", fs, facts, code)
    st.download_button("⬇️ Download report (.md)", md, "ml_pipeline_doctor_report.md", use_container_width=True)
    st.download_button("⬇️ Download data (.json)", json.dumps(dict(facts=facts, findings=fs), default=str, indent=2), "ml_pipeline_doctor_report.json", use_container_width=True)
    st.download_button("⬇️ Download corrected pipeline (.py)", code, "corrected_pipeline.py", use_container_width=True)
    st.markdown(md)
