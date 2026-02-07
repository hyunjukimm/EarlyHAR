# ecm_template.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering


# -----------------------------
# Metrics: Earliness + HM
# -----------------------------
def earliness_ratio(tau: int, T: int) -> float:
    return float(tau) / float(T)

def harmonic_mean_f1_earliness(f1: float, el: float, eps: float = 1e-12) -> float:
    # 흔히 Early classification에서 (F1)와 (1-EL)을 tradeoff로 보고 HM을 씁니다.
    # HM = 2 * F1 * (1-EL) / (F1 + (1-EL))
    a = f1
    b = 1.0 - el
    return (2.0 * a * b) / (a + b + eps)


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class MTDSample:
    # components[i] shape: (Ti,) or (Ti, d_i) but 여기서는 1D를 가정
    components: List[np.ndarray]
    label: int

@dataclass
class ECMModel:
    # per-component GP classifier
    gps: List[GaussianProcessClassifier]
    # per-component, per-class MRD table: mrd[i][k] = required prefix length for class k
    mrd: List[np.ndarray]  # shape (n_components, n_classes)
    # sampling rates aligned with components order
    lambdas: np.ndarray
    # maximum full length (in "fastest component ticks")
    T: int


# -----------------------------
# Featureization
# -----------------------------
def pad_to_T(x: np.ndarray, T: int) -> np.ndarray:
    # x: (t,)
    if x.shape[0] >= T:
        return x[:T].astype(np.float32)
    out = np.zeros((T,), dtype=np.float32)
    out[: x.shape[0]] = x.astype(np.float32)
    return out

def featureize_prefix(component_ts: np.ndarray, t: int, T: int) -> np.ndarray:
    """
    논문은 prefix length t로 posterior를 계산합니다.
    GP는 입력 차원이 고정이어야 하므로, prefix를 T까지 0-padding한 벡터를 기본값으로 둡니다.
    """
    prefix = component_ts[:t]
    feat = pad_to_T(prefix, T)
    return feat


# -----------------------------
# Phase 1: MRD estimation (Algorithm 1)
# -----------------------------
def fit_gp_classifier(X: np.ndarray, y: np.ndarray) -> GaussianProcessClassifier:
    # 안정적인 기본 커널
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e3))
    gp = GaussianProcessClassifier(kernel=kernel, random_state=0, max_iter_predict=100)
    gp.fit(X, y)
    return gp

def top1_top2_gap(proba: np.ndarray) -> Tuple[float, float, float]:
    # proba: (n_classes,)
    s = np.sort(proba)[::-1]
    top1 = float(s[0])
    top2 = float(s[1]) if s.shape[0] > 1 else 0.0
    delta = top1 - top2
    return top1, top2, delta

def estimate_mrd_for_component(
    samples: List[MTDSample],
    comp_idx: int,
    n_classes: int,
    T: int,
    alpha: float,
    use_clustering: bool = False,
    n_clusters: int = 2,
) -> Tuple[GaussianProcessClassifier, np.ndarray]:
    """
    Algorithm 1 요지를 따라 MRD_i,k를 뽑습니다. :contentReference[oaicite:10]{index=10}
    - 각 샘플 j에 대해 MRD(C^j_i) 를 구하고
    - class-wise MRD_i,k는 그 class 샘플들의 MRD 최대(보수적)로 둡니다.
    - 논문의 hierarchical clustering 기반 natural group은 옵션 처리합니다.
    """
    # Build full-length training set for GP
    X_full = []
    y_full = []
    comp_series = []
    labels = []
    for s in samples:
        ts = s.components[comp_idx]
        X_full.append(featureize_prefix(ts, T, T))
        y_full.append(s.label)
        comp_series.append(ts)
        labels.append(s.label)
    X_full = np.stack(X_full, axis=0)
    y_full = np.asarray(y_full, dtype=int)

    gp = fit_gp_classifier(X_full, y_full)

    # Precompute full-length proba per sample
    proba_T = gp.predict_proba(X_full)  # (N, C)

    # MRD per sample (init as T)
    N = len(samples)
    mrd_sample = np.full((N,), T, dtype=int)

    for j in range(N):
        top1_T, _, delta_T = top1_top2_gap(proba_T[j])

        ts = samples[j].components[comp_idx]
        for t in range(1, T + 1):
            x_t = featureize_prefix(ts, t, T)[None, :]
            proba_t = gp.predict_proba(x_t)[0]
            top1_t, _, delta_t = top1_top2_gap(proba_t)

            # Algorithm 1 condition (line 9) :contentReference[oaicite:11]{index=11}
            if (delta_T <= delta_t) and (alpha * top1_T <= top1_t):
                mrd_sample[j] = t
                break

    # class-wise MRD: conservative max over samples of each class
    mrd_class = np.ones((n_classes,), dtype=int)
    for k in range(n_classes):
        idx = np.where(y_full == k)[0]
        if idx.size == 0:
            mrd_class[k] = T
        else:
            if not use_clustering:
                mrd_class[k] = int(np.max(mrd_sample[idx]))
            else:
                # optional: clustering within class, take max MRD per cluster then overall max
                Xc = X_full[idx]
                # AgglomerativeClustering expects 2D features
                clusterer = AgglomerativeClustering(n_clusters=min(n_clusters, idx.size))
                cl = clusterer.fit_predict(Xc)
                mrd_k = []
                for c_id in np.unique(cl):
                    sub = idx[cl == c_id]
                    mrd_k.append(int(np.max(mrd_sample[sub])))
                mrd_class[k] = int(np.max(mrd_k))

    return gp, mrd_class


def train_ecm(
    samples: List[MTDSample],
    lambdas: List[float],
    n_classes: int,
    T: int,
    alpha: float = 0.9,
    use_clustering: bool = False,
) -> ECMModel:
    lambdas = np.asarray(lambdas, dtype=float)

    # sort by non-increasing lambdas (paper assumption) :contentReference[oaicite:12]{index=12}
    order = np.argsort(-lambdas)
    lambdas_sorted = lambdas[order]

    # reorder components accordingly
    samples_sorted = []
    for s in samples:
        comps = [s.components[i] for i in order]
        samples_sorted.append(MTDSample(components=comps, label=s.label))

    n_components = len(lambdas_sorted)
    gps: List[GaussianProcessClassifier] = []
    mrd: List[np.ndarray] = []

    for i in range(n_components):
        gp_i, mrd_i = estimate_mrd_for_component(
            samples_sorted,
            comp_idx=i,
            n_classes=n_classes,
            T=T,
            alpha=alpha,
            use_clustering=use_clustering,
        )
        gps.append(gp_i)
        mrd.append(mrd_i)

    return ECMModel(gps=gps, mrd=mrd, lambdas=lambdas_sorted, T=T)


# -----------------------------
# Phase 2: Divide-and-Conquer inference
# -----------------------------
@dataclass
class TreeNode:
    idxs: List[int]                 # component indices in this node
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

def build_dc_tree(comp_idxs: List[int], lambdas: np.ndarray) -> TreeNode:
    if len(comp_idxs) == 1:
        return TreeNode(idxs=comp_idxs)

    # split by maximum gap in consecutive lambdas as in paper description :contentReference[oaicite:13]{index=13}
    ls = lambdas[comp_idxs]
    gaps = np.abs(ls[:-1] - ls[1:])
    if np.allclose(gaps, gaps[0]):
        mid = len(comp_idxs) // 2
        left_idxs = comp_idxs[:mid]
        right_idxs = comp_idxs[mid:]
    else:
        split_pos = int(np.argmax(gaps)) + 1
        left_idxs = comp_idxs[:split_pos]
        right_idxs = comp_idxs[split_pos:]

    node = TreeNode(idxs=comp_idxs)
    node.left = build_dc_tree(left_idxs, lambdas)
    node.right = build_dc_tree(right_idxs, lambdas)
    return node

def predict_component_label(
    model: ECMModel,
    comp_ts: np.ndarray,
    comp_i: int,
    m_i: int,
) -> int:
    x = featureize_prefix(comp_ts, max(1, m_i), model.T)[None, :]
    proba = model.gps[comp_i].predict_proba(x)[0]
    return int(np.argmax(proba))

def simulate_stream_predict(
    model: ECMModel,
    sample: MTDSample,
) -> Tuple[int, int]:
    """
    returns: (pred_label, tau)
    tau는 'fastest component ticks' 기준의 stopping time으로 둡니다.
    각 tick t에서 component i의 관측 길이는 m_i = floor(lambda_i * t) 로 근사합니다.
    """
    n = len(sample.components)
    root = build_dc_tree(list(range(n)), model.lambdas)

    def merge(node: TreeNode, t: int) -> Tuple[int, int]:
        """
        returns: (label, required_t)
        required_t는 이 subtree 결론을 내기 위해 필요한 최소 tick.
        """
        if node.left is None and node.right is None:
            i = node.idxs[0]
            m_i = int(np.floor(model.lambdas[i] * t))
            m_i = max(1, min(model.T, m_i))
            lab = predict_component_label(model, sample.components[i], i, m_i)
            # leaf 자체는 현재 t에서 판단, 추가 대기 요구는 0
            return lab, 0

        lab_l, req_l = merge(node.left, t)
        lab_r, req_r = merge(node.right, t)

        if lab_l == lab_r:
            return lab_l, max(req_l, req_r)

        # labels differ, apply paper's "wait until right group can predict left label" idea :contentReference[oaicite:14]{index=14}
        # 간단화: right subtree의 "가장 느린(마지막)" 컴포넌트를 representative로 사용
        right_last = node.right.idxs[-1]
        need_m = int(model.mrd[right_last][lab_l])  # MRD of class lab_l in that component
        # need tick such that floor(lambda * tick) >= need_m
        lam = float(model.lambdas[right_last])
        need_t = int(np.ceil(need_m / max(lam, 1e-12)))

        if t >= need_t:
            # enough data, accept right's current label as in description (assign last component of y)
            return lab_r, max(req_l, req_r)
        else:
            # must wait more
            return lab_r, max(req_l, req_r, need_t)

    # stream over t=1..T
    for t in range(1, model.T + 1):
        lab, need_t = merge(root, t)
        if need_t <= t:
            return lab, t

    # fallback: use full length
    lab_full, _ = merge(root, model.T)
    return lab_full, model.T


# -----------------------------
# Experiment loop template with F1, EL, HM
# -----------------------------
def evaluate_ecm(
    model: ECMModel,
    test_samples: List[MTDSample],
) -> Dict[str, float]:
    y_true = []
    y_pred = []
    taus = []
    for s in test_samples:
        pred, tau = simulate_stream_predict(model, s)
        y_true.append(s.label)
        y_pred.append(pred)
        taus.append(tau)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    taus = np.asarray(taus, dtype=int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    el = float(np.mean([earliness_ratio(int(t), model.T) for t in taus]))
    hm = float(harmonic_mean_f1_earliness(macro_f1, el))

    return {
        "macro_f1": macro_f1,
        "earliness": el,
        "hm": hm,
        "avg_tau": float(np.mean(taus)),
    }
