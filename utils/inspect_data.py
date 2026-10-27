"""
Kiểm tra dữ liệu PEMS (train/val/test.npz) MỘT LẦN, lưu báo cáo vào cache.

Cách dùng:
    python -m utils.inspect_data --data_dir data/PEMS08            # tạo báo cáo đầy đủ + hình
    python -m utils.inspect_data --data_dir data/PEMS08 --force    # bắt buộc kiểm tra lại

Trong code train/evaluate chỉ cần gọi:
    from utils.inspect_data import ensure_report
    ensure_report("data/PEMS08")   # in 1 dòng tóm tắt nếu dữ liệu không đổi

Báo cáo được lưu ở reports/<TÊN_DATASET>/ gồm: report.json, report.md và các hình .png.
Cache được khóa bằng "dấu vân tay" (kích thước + thời gian sửa file .npz), nên khi bạn
thay dữ liệu thì báo cáo tự được tạo lại.
"""
import argparse
import hashlib
import json
import os
import time
import warnings as _pywarn

import numpy as np

from config import DATA_DIR

REPORT_VERSION = 1
SPLITS = ("train", "val", "test")

# Thư mục gốc project (= thư mục cha của utils/). Đường dẫn tương đối luôn tính từ đây,
# nên chạy script từ thư mục nào (terminal, PyCharm, VS Code) cũng đúng.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Số liệu trong Table 2 của bài báo STIDGCN để đối chiếu
PAPER_STATS = {
    "PEMS03": dict(nodes=358, timesteps=26208, missing_pct=0.672),
    "PEMS04": dict(nodes=307, timesteps=16992, missing_pct=3.182),
    "PEMS07": dict(nodes=883, timesteps=28224, missing_pct=0.452),
    "PEMS08": dict(nodes=170, timesteps=17856, missing_pct=0.696),
}


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------
def _resolve(path):
    """Đổi đường dẫn tương đối thành đường dẫn tính từ thư mục gốc project."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _check_files(data_dir):
    """Báo lỗi dễ hiểu nếu thiếu thư mục hoặc thiếu file .npz."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}\n"
                                f"Thư mục gốc project đang là: {PROJECT_ROOT}")
    missing = [f"{s}.npz" for s in SPLITS if not os.path.isfile(os.path.join(data_dir, f"{s}.npz"))]
    if missing:
        raise FileNotFoundError(f"Thư mục {data_dir} thiếu {missing}. "
                                f"Các file đang có: {sorted(os.listdir(data_dir))}")


def _fingerprint(data_dir):
    h = hashlib.sha1(f"v{REPORT_VERSION}".encode())
    for s in SPLITS:
        p = os.path.join(data_dir, f"{s}.npz")
        st = os.stat(p)
        h.update(f"{s}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


def _longest_run(mask_1d):
    """Độ dài đoạn True liên tiếp dài nhất trong mảng bool 1 chiều."""
    if not mask_1d.any():
        return 0
    padded = np.concatenate([[0], mask_1d.astype(np.int8), [0]])
    d = np.diff(padded)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((ends - starts).max())


def _longest_stuck(series_1d):
    """Đoạn dài nhất mà giá trị KHÁC 0 bị lặp lại y hệt (cảm biến 'đứng hình')."""
    same = np.concatenate([[False], series_1d[1:] == series_1d[:-1]])
    same &= series_1d != 0
    return _longest_run(same) + (1 if same.any() else 0)


def _reconstruct(x, y):
    """Ghép các cửa sổ trượt (stride=1) lại thành chuỗi gốc liên tục.

    x: (S, T_in, N, F), y: (S, T_out, N, Fy)
    flow: (S + T_in + T_out - 1, N);  feats: (S + T_in - 1, N, F)
    """
    flow = np.concatenate([x[:, 0, :, 0], x[-1, 1:, :, 0], y[-1, :, :, 0]], axis=0)
    feats = np.concatenate([x[:, 0], x[-1, 1:]], axis=0)
    return flow, feats


def _continuity(x, y, n_check=1000, seed=0):
    """Tỷ lệ cặp mẫu liền kề thỏa x[i+1] = x[i] dịch 1 bước (dữ liệu chưa bị xáo trộn)."""
    S = x.shape[0]
    if S < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(S - 1, size=min(n_check, S - 1), replace=False)
    ok_x = np.array([np.allclose(x[i, 1:, :, 0], x[i + 1, :-1, :, 0]) for i in idx])
    ok_y = np.array([np.allclose(y[i, 0, :, 0], x[i + 1, -1, :, 0]) for i in idx])
    return float((ok_x & ok_y).mean())


# ----------------------------------------------------------------------------
# Phân tích từng tập
# ----------------------------------------------------------------------------
def _analyze_split(name, x, y, warnings):
    r = {"x_shape": list(x.shape), "y_shape": list(y.shape),
         "x_dtype": str(x.dtype), "y_dtype": str(y.dtype)}
    S, T_in, N, F = x.shape

    if y.shape[0] != S:
        warnings.append(("ERROR", f"[{name}] số mẫu x ({S}) khác y ({y.shape[0]})"))

    # 1. NaN / Inf theo từng đặc trưng
    r["nan_per_feature"] = [int(np.isnan(x[..., f]).sum()) for f in range(F)]
    r["inf_per_feature"] = [int(np.isinf(x[..., f]).sum()) for f in range(F)]
    r["nan_y"] = int(np.isnan(y).sum())
    if sum(r["nan_per_feature"]) + r["nan_y"] > 0:
        warnings.append(("ERROR", f"[{name}] có NaN: x={r['nan_per_feature']}, y={r['nan_y']}"))

    # 2. Kiểm tra cửa sổ trượt có liên tục không (để ghép lại chuỗi gốc)
    r["window_continuity"] = _continuity(x, y)
    continuous = r["window_continuity"] > 0.99
    if not continuous:
        warnings.append(("WARN", f"[{name}] cửa sổ không liên tục ({r['window_continuity']:.1%}) "
                                 "→ dữ liệu đã bị xáo trộn hoặc stride≠1; thống kê tính trên cửa sổ chồng lấn"))

    if continuous:
        flow, feats = _reconstruct(x, y)
    else:
        flow, feats = x[..., 0].reshape(-1, N), x.reshape(-1, N, F)
    r["series_len"] = int(flow.shape[0])

    # 3. Thống kê lưu lượng (feature 0). PEMS mã hóa dữ liệu thiếu bằng 0, KHÔNG phải NaN
    zero = flow == 0
    nz = flow[~zero]
    r["flow"] = {
        "min": float(flow.min()), "max": float(flow.max()),
        "mean_all": float(flow.mean()), "std_all": float(flow.std()),
        "mean_nonzero": float(nz.mean()) if nz.size else None,
        "std_nonzero": float(nz.std()) if nz.size else None,
        "p01_p50_p99": [float(v) for v in np.percentile(nz, [1, 50, 99])] if nz.size else None,
        "zero_pct": float(zero.mean() * 100),
        "negative_count": int((flow < 0).sum()),
    }
    if r["flow"]["negative_count"] > 0:
        warnings.append(("ERROR", f"[{name}] có {r['flow']['negative_count']} giá trị lưu lượng âm"))

    # Ngoại lai theo IQR (chỉ tính trên giá trị khác 0)
    if nz.size:
        q1, q3 = np.percentile(nz, [25, 75])
        r["flow"]["outlier_pct_iqr3"] = float((nz > q3 + 3 * (q3 - q1)).mean() * 100)

    # 4. Theo từng nút cảm biến
    zero_node = zero.mean(axis=0)
    std_node = flow.std(axis=0)
    top = np.argsort(-zero_node)[:5]
    r["nodes"] = {
        "top5_missing": [{"node": int(i), "zero_pct": float(zero_node[i] * 100)} for i in top],
        "n_nodes_missing_gt_50pct": int((zero_node > 0.5).sum()),
        "constant_nodes": [int(i) for i in np.where(std_node == 0)[0]],
    }
    if continuous:
        gap = np.array([_longest_run(zero[:, n]) for n in range(N)])
        stuck = np.array([_longest_stuck(flow[:, n]) for n in range(N)])
        r["nodes"]["longest_missing_gap_steps"] = int(gap.max())
        r["nodes"]["node_longest_gap"] = int(gap.argmax())
        r["nodes"]["longest_stuck_steps"] = int(stuck.max())
        r["nodes"]["node_longest_stuck"] = int(stuck.argmax())
        if stuck.max() >= 12:
            warnings.append(("WARN", f"[{name}] nút {int(stuck.argmax())} lặp nguyên một giá trị "
                                     f"{int(stuck.max())} bước liên tiếp (≥1 giờ) → nghi cảm biến lỗi"))
    if r["nodes"]["constant_nodes"]:
        warnings.append(("WARN", f"[{name}] nút có giá trị không đổi: {r['nodes']['constant_nodes']}"))

    # 5. Đặc trưng thời gian (feature 1 = thời điểm trong ngày, feature 2 = thứ trong tuần)
    if F >= 2 and continuous:
        tod = feats[:, 0, 1]
        r["time_of_day_range"] = [float(tod.min()), float(tod.max())]
        d = np.mod(np.diff(tod), 1.0)
        step = float(np.median(d)) if d.size else 0
        r["steps_per_day"] = int(round(1 / step)) if step > 0 else None
        r["time_gaps"] = int((np.abs(d - step) > 1e-4).sum())
        if r["time_gaps"]:
            warnings.append(("WARN", f"[{name}] có {r['time_gaps']} chỗ nhảy cóc thời gian"))
        if tod.min() < 0 or tod.max() >= 1:
            warnings.append(("ERROR", f"[{name}] time-of-day nằm ngoài [0, 1)"))
    if F >= 3:
        dow = feats[:, 0, 2]
        r["day_of_week_values"] = sorted(int(v) for v in np.unique(dow))
        if not np.all(np.isin(dow, np.arange(7))):
            warnings.append(("ERROR", f"[{name}] day-of-week không phải số nguyên 0..6"))

    r["_flow_series"] = flow  # dùng tạm để vẽ hình, không ghi ra json
    r["_tod_series"] = feats[:, 0, 1] if (F >= 2 and continuous) else None
    return r


# ----------------------------------------------------------------------------
# Vẽ hình
# ----------------------------------------------------------------------------
def _plots(rep, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    train = rep["splits"]["train"]
    flow = train["_flow_series"]

    # Histogram lưu lượng của 3 tập
    plt.figure(figsize=(8, 4))
    for s in SPLITS:
        f = rep["splits"][s]["_flow_series"]
        plt.hist(f[f > 0].ravel(), bins=100, alpha=0.5, density=True, label=s)
    plt.xlabel("Lưu lượng (khác 0)")
    plt.ylabel("Mật độ")
    plt.title("Phân phối lưu lượng theo tập")
    plt.legend()
    plt.tight_layout()
    figs["hist"] = "flow_hist.png"
    plt.savefig(os.path.join(out_dir, figs["hist"]), dpi=150)
    plt.close()

    # Hồ sơ trung bình theo giờ trong ngày
    spd = train.get("steps_per_day")
    if spd and train["_tod_series"] is not None:
        plt.figure(figsize=(8, 4))
        for s in SPLITS:
            f, tod = rep["splits"][s]["_flow_series"], rep["splits"][s]["_tod_series"]
            n = len(tod)
            slot = np.round(tod * spd).astype(int) % spd
            f = np.where(f[:n] == 0, np.nan, f[:n]).mean(axis=1)
            with _pywarn.catch_warnings():
                _pywarn.simplefilter("ignore", category=RuntimeWarning)
                prof = np.array([np.nanmean(f[slot == k]) if (slot == k).any() else np.nan
                                 for k in range(spd)])
            plt.plot(np.arange(spd) * 24 / spd, prof, label=s)
        plt.xlabel("Giờ trong ngày")
        plt.ylabel("Lưu lượng TB toàn mạng")
        plt.title("Hồ sơ lưu lượng trung bình theo giờ")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        figs["daily"] = "daily_profile.png"
        plt.savefig(os.path.join(out_dir, figs["daily"]), dpi=150)
        plt.close()

    # Heatmap tỷ lệ thiếu (giá trị 0) theo nút × ngày trên tập train
    spd = spd or 288
    n_days = flow.shape[0] // spd
    if n_days >= 2:
        z = (flow[: n_days * spd] == 0).reshape(n_days, spd, -1).mean(axis=1).T * 100
        plt.figure(figsize=(10, 5))
        plt.imshow(z, aspect="auto", cmap="Reds", vmin=0, vmax=max(5, z.max()))
        plt.colorbar(label="% giá trị = 0")
        plt.xlabel("Ngày (tập train)")
        plt.ylabel("Nút cảm biến")
        plt.title("Bản đồ dữ liệu thiếu theo nút × ngày")
        plt.tight_layout()
        figs["missing"] = "missing_heatmap.png"
        plt.savefig(os.path.join(out_dir, figs["missing"]), dpi=150)
        plt.close()

    # 1 tuần của nút điển hình và nút thiếu nhiều nhất
    worst = train["nodes"]["top5_missing"][0]["node"]
    typical = int(np.argmin(np.abs(flow.mean(0) - np.median(flow.mean(0)))))
    week = min(7 * spd, flow.shape[0])
    plt.figure(figsize=(12, 4))
    plt.plot(flow[:week, typical], lw=0.8, label=f"Nút điển hình #{typical}")
    plt.plot(flow[:week, worst], lw=0.8, alpha=0.8, label=f"Nút thiếu nhiều nhất #{worst}")
    plt.xlabel("Bước thời gian")
    plt.ylabel("Lưu lượng")
    plt.title("Tuần đầu tiên của tập train")
    plt.legend()
    plt.tight_layout()
    figs["week"] = "node_week.png"
    plt.savefig(os.path.join(out_dir, figs["week"]), dpi=150)
    plt.close()
    return figs


# ----------------------------------------------------------------------------
# Tạo báo cáo
# ----------------------------------------------------------------------------
def build_report(data_dir, name=None, out_root="reports", make_plots=True):
    name = name or os.path.basename(os.path.normpath(data_dir))
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    warnings = []
    rep = {"dataset": name, "data_dir": data_dir, "fingerprint": _fingerprint(data_dir),
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "splits": {}}

    for s in SPLITS:
        d = np.load(os.path.join(data_dir, f"{s}.npz"))
        rep["splits"][s] = _analyze_split(s, d["x"], d["y"], warnings)
        rep["splits"][s]["npz_keys"] = list(d.keys())

    tr, va, te = (rep["splits"][s] for s in SPLITS)

    # Số nút, số đặc trưng phải khớp giữa các tập
    shapes = {s: rep["splits"][s]["x_shape"][1:] for s in SPLITS}
    if len({tuple(v) for v in shapes.values()}) > 1:
        warnings.append(("ERROR", f"Kích thước (T_in, N, F) khác nhau giữa các tập: {shapes}"))

    # Tỷ lệ chia và tổng số bước thời gian
    n = {s: rep["splits"][s]["x_shape"][0] for s in SPLITS}
    total = sum(n.values())
    T_in, T_out = tr["x_shape"][1], tr["y_shape"][1]
    rep["split_ratio"] = {s: round(n[s] / total, 4) for s in SPLITS}
    rep["estimated_total_timesteps"] = total + T_in + T_out - 1

    # Lệch phân phối giữa các tập (so với train)
    m_tr = tr["flow"]["mean_nonzero"]
    rep["mean_shift_vs_train_pct"] = {s: round((rep["splits"][s]["flow"]["mean_nonzero"] - m_tr) / m_tr * 100, 2)
                                      for s in ("val", "test")}
    for s, v in rep["mean_shift_vs_train_pct"].items():
        if abs(v) > 15:
            warnings.append(("WARN", f"Trung bình lưu lượng tập {s} lệch {v:+.1f}% so với train"))

    # Thông số chuẩn hóa (giống cách cả code tác giả và code của bạn tính: trên x_train, gồm cả số 0)
    d = np.load(os.path.join(data_dir, "train.npz"))
    rep["scaler"] = {"mean": float(d["x"][..., 0].mean()), "std": float(d["x"][..., 0].std())}

    # Tỷ lệ thiếu tổng thể (trên chuỗi ghép lại, không bị đếm trùng do cửa sổ chồng lấn)
    zeros = sum((rep["splits"][s]["_flow_series"] == 0).sum() for s in SPLITS)
    cells = sum(rep["splits"][s]["_flow_series"].size for s in SPLITS)
    rep["overall_zero_pct"] = float(zeros / cells * 100)

    # Đối chiếu với bài báo
    paper = PAPER_STATS.get(name.upper())
    if paper:
        rep["paper"] = paper
        N = tr["x_shape"][2]
        if N != paper["nodes"]:
            warnings.append(("ERROR", f"Số nút {N} ≠ bài báo ({paper['nodes']})"))
        if abs(rep["estimated_total_timesteps"] - paper["timesteps"]) > T_in + T_out:
            warnings.append(("WARN", f"Tổng số bước ước lượng {rep['estimated_total_timesteps']} "
                                     f"≠ bài báo ({paper['timesteps']})"))

    if any(abs(rep["split_ratio"][s] - r) > 0.02 for s, r in zip(SPLITS, (0.6, 0.2, 0.2))):
        warnings.append(("WARN", f"Tỷ lệ chia {rep['split_ratio']} khác 6:2:2 như bài báo"))

    if make_plots:
        rep["figures"] = _plots(rep, out_dir)

    for s in SPLITS:
        rep["splits"][s].pop("_flow_series")
        rep["splits"][s].pop("_tod_series")
    rep["warnings"] = [{"level": l, "msg": m} for l, m in warnings]
    rep["elapsed_sec"] = round(time.time() - t0, 2)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(_to_markdown(rep))
    return rep


def _to_markdown(rep):
    L = [f"# Báo cáo dữ liệu {rep['dataset']}", "",
         f"Tạo lúc {rep['created']} · fingerprint `{rep['fingerprint']}`", "",
         "## Tổng quan", "",
         "| Tập | x shape | y shape | Độ dài chuỗi | % giá trị 0 | TB (≠0) | Std (≠0) | Liên tục |",
         "|---|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        r = rep["splits"][s]
        L.append(f"| {s} | {tuple(r['x_shape'])} | {tuple(r['y_shape'])} | {r['series_len']} | "
                 f"{r['flow']['zero_pct']:.3f}% | {r['flow']['mean_nonzero']:.2f} | "
                 f"{r['flow']['std_nonzero']:.2f} | {r['window_continuity']:.0%} |")
    L += ["", f"- Tỷ lệ chia: {rep['split_ratio']}",
          f"- Tổng số bước thời gian ước lượng: {rep['estimated_total_timesteps']}",
          f"- Tỷ lệ giá trị 0 toàn bộ: {rep['overall_zero_pct']:.3f}%",
          f"- Scaler (train, gồm cả 0): mean={rep['scaler']['mean']:.3f}, std={rep['scaler']['std']:.3f}"]
    if "paper" in rep:
        p = rep["paper"]
        L.append(f"- Bài báo (Table 2): {p['nodes']} nút, {p['timesteps']} bước, missing {p['missing_pct']}%")
    L += ["", "## Cảnh báo", ""]
    L += [f"- **{w['level']}**: {w['msg']}" for w in rep["warnings"]] or ["Không có cảnh báo."]
    if rep.get("figures"):
        L += ["", "## Hình", ""] + [f"![{k}]({v})" for k, v in rep["figures"].items()]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------
# API dùng trong train/evaluate
# ----------------------------------------------------------------------------
def ensure_report(data_dir, name=None, out_root="reports", force=False, strict=True, verbose=True):
    """Dùng lại báo cáo cũ nếu dữ liệu không đổi; ngược lại kiểm tra lại từ đầu.

    strict=True: dừng chương trình nếu có lỗi mức ERROR (NaN, sai shape...).
    """
    data_dir, out_root = _resolve(data_dir), _resolve(out_root)
    _check_files(data_dir)
    name = name or os.path.basename(os.path.normpath(data_dir))
    path = os.path.join(out_root, name, "report.json")
    rep, cached = None, False
    if not force and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("fingerprint") == _fingerprint(data_dir):
            rep, cached = old, True
    if rep is None:
        if verbose:
            print(f"[data] Đang kiểm tra {name} (lần đầu hoặc dữ liệu đã thay đổi)...")
        rep = build_report(data_dir, name, out_root)

    if verbose:
        print_summary(rep, cached)
    errors = [w["msg"] for w in rep["warnings"] if w["level"] == "ERROR"]
    if strict and errors:
        raise ValueError("Dữ liệu có lỗi:\n  " + "\n  ".join(errors))
    return rep


def print_summary(rep, cached=False):
    tr = rep["splits"]["train"]
    shp = " | ".join(f"{s} {tuple(rep['splits'][s]['x_shape'])}" for s in SPLITS)
    tag = "cache" if cached else "mới tạo"
    n_warn = len(rep["warnings"])
    print(f"[data] {rep['dataset']} ({tag}, {rep['fingerprint']}) {shp}")
    print(f"[data] missing(=0) {rep['overall_zero_pct']:.3f}%"
          + (f" (bài báo {rep['paper']['missing_pct']}%)" if "paper" in rep else "")
          + f" · NaN {sum(tr['nan_per_feature'])} · scaler μ={rep['scaler']['mean']:.2f} "
          f"σ={rep['scaler']['std']:.2f} · {n_warn} cảnh báo"
          + (f" → xem reports/{rep['dataset']}/report.md" if n_warn else ""))
    for w in rep["warnings"]:
        print(f"[data]   {w['level']}: {w['msg']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--out_root", default="reports")
    ap.add_argument("--force", action="store_true", help="bỏ qua cache, kiểm tra lại")
    args = ap.parse_args()
    ensure_report(args.data_dir, out_root=args.out_root, force=args.force, strict=False)