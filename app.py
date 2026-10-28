"""Chạy từ thư mục dự án: python -m streamlit run app.py"""

import io
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src.ibft_utils import (
    AMOUNT, FEATURES, TIME, TEMPLATE_COLUMNS, X_COLUMNS,
    clean_transactions, csv_bytes, display_transactions, filter_transactions,
    predict_frame, predict_upload, read_transactions,
)

ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / "output" / "prepared"
MODEL = ROOT / "output" / "kmeans"
COMPARISON = ROOT / "output" / "comparison" / "comparison_final.csv"
COLORS = ["#2563EB", "#EA8A23", "#199B77", "#9856C5", "#D55B70"]

st.set_page_config(page_title="IBFT Explorer", page_icon="◈", layout="wide")
st.markdown("""<style>
    .stApp {background: #F7F9FC; color: #17243B;}
    .block-container {max-width: 1240px; padding-top: 2.5rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -1.5px; font-weight: 750 !important;}
    h2, h3 {letter-spacing: -.4px;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #DFE5EE;
        border-radius: 14px; padding: 16px 20px;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
    .eyebrow {color: #2563EB; font-size: .78rem; font-weight: 700; letter-spacing: 2px;}
    footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)


def file_stamp(paths):
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


@st.cache_resource(show_spinner=False)
def load_assets(stamp):
    preprocessor = joblib.load(PREPARED / "preprocessor.joblib")
    model = joblib.load(MODEL / "model.joblib")
    metrics = json.loads((MODEL / "metrics.json").read_text(encoding="utf-8"))
    if list(preprocessor.feature_names_in_) != FEATURES or model.n_features_in_ != len(FEATURES):
        raise ValueError("Đặc trưng mô hình không khớp ứng dụng. Hãy kiểm tra lại notebook 01 và K-Means.")
    if model.n_clusters != metrics["best_k"] or model.random_state != metrics["best_seed"]:
        raise ValueError("Model và metrics.json không cùng lần chạy. Hãy chạy lại notebook K-Means.")
    prepared = pd.read_parquet(PREPARED / "model_input.parquet")
    pca = PCA(n_components=2, svd_solver="full").fit(prepared.loc[prepared["split"].eq("train"), X_COLUMNS].to_numpy())
    return preprocessor, model, pca, metrics


@st.cache_data(show_spinner=False)
def existing_result(stamp):
    _, model, pca, _ = load_assets(stamp)
    prepared = pd.read_parquet(PREPARED / "model_input.parquet")
    lookup = pd.read_parquet(PREPARED / "transaction_lookup.parquet")
    features = pd.read_parquet(PREPARED / "features_unscaled.parquet")
    data = prepared[["transaction_id"]].merge(lookup, on="transaction_id", validate="one_to_one")
    data = data.merge(features[["transaction_id", "log_sender_tx_count_7d", "is_new_receiver"]],
                      on="transaction_id", validate="one_to_one")
    if len(data) != len(prepared):
        raise ValueError("Các file prepared không đủ giao dịch tương ứng. Hãy chạy lại notebook 01.")
    data["sender_tx_count_7d"] = np.expm1(data["log_sender_tx_count_7d"]).round().astype(int)
    result = predict_frame(data, prepared[X_COLUMNS], model, pca)
    return result, {"received": len(result), "invalid": 0, "duplicates": 0, "warmup": 0,
                    "scored": len(result), "missing_amount": int(data[AMOUNT].isna().sum())}


@st.cache_data(show_spinner=False)
def load_history(stamp):
    tables = [read_transactions(Path(path), Path(path).name) for path, _, _ in stamp]
    history, _ = clean_transactions(pd.concat(tables, ignore_index=True))
    return history


def number(value):
    return f"{value:,.0f}".replace(",", ".")


st.markdown('<div class="eyebrow">PHÂN TÍCH GIAO DỊCH · LOCAL</div>', unsafe_allow_html=True)
st.title("IBFT Explorer")
st.write("Từ giao dịch đến nhóm hành vi. Khám phá từng nhóm và tra cứu kết quả trong một màn hình.")

try:
    required = [PREPARED / name for name in ["preprocessor.joblib", "model_input.parquet",
                "transaction_lookup.parquet", "features_unscaled.parquet", "preparation_info.json"]]
    required += [MODEL / "model.joblib", MODEL / "metrics.json", COMPARISON]
    stamp = file_stamp(required)
    preprocessor, model, pca, metrics = load_assets(stamp)
    preparation = json.loads((PREPARED / "preparation_info.json").read_text(encoding="utf-8"))
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Chưa đọc được dữ liệu hoặc mô hình: {exc}")
    st.info("Chạy notebook 01, các notebook mô hình và notebook 02 để tạo đủ kết quả trong output/.")
    st.stop()

st.caption(f"K-Means · {model.n_clusters} cụm · seed {metrics['best_seed']} · Dùng mô hình đã huấn luyện")
with st.container(border=True):
    st.subheader("01  Chọn dữ liệu")
    mode = st.radio("Nguồn giao dịch", ["Dữ liệu có sẵn", "Tải file mới"], horizontal=True, label_visibility="collapsed")
    uploads, use_history = [], False
    history_paths = [ROOT / "data" / item["file"] for item in preparation["input_files"]]
    if mode == "Tải file mới":
        uploads = st.file_uploader("Excel (.xlsx) hoặc CSV (.csv)", type=["xlsx", "csv"],
                                   accept_multiple_files=True, max_upload_size=25)
        has_history = bool(history_paths) and all(path.is_file() for path in history_paths)
        use_history = st.checkbox("Dùng thêm lịch sử có sẵn trong data/", value=has_history, disabled=not has_history)
        st.caption("File được xử lý trong bộ nhớ, không ghi vào thư mục dự án. Giới hạn 100.000 dòng mỗi lần.")
        with st.expander("Định dạng file và lịch sử cần có"):
            st.write("Dùng Excel cùng cấu trúc dữ liệu gốc (sheet IBFT), hoặc CSV có các cột dưới đây.")
            st.code(", ".join(TEMPLATE_COLUMNS), language=None)
            st.write("Thời gian: YYYY-MM-DD HH:MM:SS, giờ Việt Nam. Số tiền là số VND, không có dấu phân cách hàng nghìn. "
                     "Mã giao dịch và tài khoản để dạng văn bản. responsecode = 0 là thành công.")
            st.write("Cần lịch sử đầy đủ trước các giao dịch cần phân nhóm. 7 ngày đầu của dữ liệu kết hợp chỉ làm lịch sử nền. "
                     "Các dòng thất bại, thiếu mã/tài khoản/thời gian, số tiền không dương và mã trùng được bỏ qua. "
                     "Số tiền trống được điền bằng bộ xử lý đã học từ train.")
            st.caption("Nếu tắt lịch sử có sẵn, hãy tải cả lịch sử và giao dịch cần phân nhóm. "
                       "Không có giao dịch trong file không đồng nghĩa tài khoản chưa từng giao dịch.")
            st.download_button("Tải mẫu CSV trống", csv_bytes(pd.DataFrame(columns=TEMPLATE_COLUMNS)),
                               "ibft_template.csv", "text/csv", on_click="ignore")
    else:
        st.write("Dùng toàn bộ giao dịch đã chuẩn bị trong output/prepared/, gồm train, validation và test.")
        st.caption("Ứng dụng chạy dự đoán lại bằng model.joblib; các chỉ số so sánh mô hình vẫn dùng kết quả đánh giá đã lưu.")

    history_stamp = file_stamp(history_paths) if use_history else ()
    signature = (mode, tuple((file.name, file.file_id) for file in uploads), history_stamp, stamp)
    if st.session_state.get("input_signature") != signature:
        st.session_state.pop("result", None)
        for key in ["cluster_filter", "date_filter", "search_filter"]:
            st.session_state.pop(key, None)
        st.session_state["input_signature"] = signature
    run = st.button("Phân nhóm giao dịch", type="primary", disabled=mode == "Tải file mới" and not uploads)
    if run or (mode == "Dữ liệu có sẵn" and "result" not in st.session_state):
        with st.spinner("Đang đọc dữ liệu và phân nhóm…"):
            try:
                if mode == "Dữ liệu có sẵn":
                    st.session_state["result"] = existing_result(stamp)
                else:
                    raw = pd.concat([read_transactions(io.BytesIO(file.getvalue()), file.name, f"upload_{i}")
                                     for i, file in enumerate(uploads)], ignore_index=True)
                    if len(raw) > 100_000:
                        raise ValueError("Mỗi lần xử lý tối đa 100.000 dòng. Hãy chia nhỏ file.")
                    history = load_history(history_stamp) if use_history else pd.DataFrame(columns=raw.columns)
                    st.session_state["result"] = predict_upload(raw, history, preprocessor, model, pca)
                    if not history.empty and raw[TIME].min() > history[TIME].max() + pd.Timedelta(days=7):
                        st.warning("File mới cách lịch sử có sẵn hơn 7 ngày. Nếu có giao dịch trong khoảng trống này, "
                                   "hãy bổ sung để đặc trưng lịch sử phản ánh đúng hoạt động.")
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Chưa thể phân nhóm: {exc}")

with st.expander("Vì sao sử dụng K-Means?"):
    comparison = pd.read_csv(COMPARISON)
    st.write("K-Means có Silhouette trung bình trên validation cao nhất và dao động nhỏ nhất giữa các seed trong kết quả hiện tại. "
             "Ứng dụng sử dụng lựa chọn này; tập test chỉ dùng để báo cáo kết quả cuối.")
    comparison = comparison[["method", "validation_silhouette_mean", "validation_silhouette_std", "silhouette"]].rename(
        columns={"method": "Mô hình", "validation_silhouette_mean": "Silhouette validation (TB)",
                 "validation_silhouette_std": "Độ lệch chuẩn", "silhouette": "Silhouette test"})
    comparison["Mô hình"] = comparison["Mô hình"].replace({"kmeans": "K-Means", "gmm": "GMM", "autoencoder_kmeans": "Autoencoder + K-Means"})
    st.dataframe(comparison.style.format(precision=5), hide_index=True)
    st.caption("Silhouette test khoảng 0,15: các nhóm còn chồng lấn. Đây là phân nhóm hành vi, không phải kết luận gian lận. "
               "Mô hình hiện được huấn luyện trên bộ dữ liệu tổng hợp của dự án.")

if "result" not in st.session_state:
    st.info("Chọn file rồi nhấn Phân nhóm giao dịch để xem kết quả.")
    st.stop()

result, stats = st.session_state["result"]
st.subheader("02  Bức tranh tổng quan")
st.caption(f"{number(stats['received'])} dòng đầu vào · {number(stats['invalid'])} dòng không hợp lệ · "
           f"{number(stats['duplicates'])} dòng trùng · {number(stats['warmup'])} dòng chỉ làm lịch sử nền")
cards = st.columns(4)
cards[0].metric("Giao dịch đã phân nhóm", number(len(result)))
cards[1].metric("Nhóm có giao dịch", result["cluster"].nunique())
median_amount = result[AMOUNT].median()
cards[2].metric("Số tiền trung vị · VND", "—" if pd.isna(median_amount) else number(median_amount))
cards[3].metric("Chuyển đến người nhận mới", f"{result['is_new_receiver'].mean():.1%}")
st.caption(f"Từ {result[TIME].min():%d/%m/%Y %H:%M} đến {result[TIME].max():%d/%m/%Y %H:%M}. "
           f"{number(stats['missing_amount'])} giao dịch thiếu số tiền: được điền khi phân nhóm, không tính vào trung vị hiển thị.")

cluster_ids = list(range(model.n_clusters))
names = [f"Cụm {cluster}" for cluster in cluster_ids]
color = alt.Color("Nhóm:N", scale=alt.Scale(domain=names, range=[COLORS[i % len(COLORS)] for i in cluster_ids]), legend=None)
counts = result["cluster"].value_counts().reindex(cluster_ids, fill_value=0)
sizes = pd.DataFrame({"Nhóm": names, "Giao dịch": counts.to_numpy()})
sizes["Tỷ lệ"] = sizes["Giao dịch"] / len(result)
left, right = st.columns([1, 1.45], gap="large")
with left:
    st.markdown("**Quy mô từng nhóm**")
    bars = alt.Chart(sizes).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("Nhóm:N", sort=names, axis=alt.Axis(labelAngle=0), title=None),
        y=alt.Y("Giao dịch:Q", title="Số giao dịch"), color=color,
        tooltip=["Nhóm:N", alt.Tooltip("Giao dịch:Q", format=","), alt.Tooltip("Tỷ lệ:Q", format=".1%")])
    st.altair_chart(bars.properties(height=310), width="stretch", alt="Số giao dịch trong từng cụm K-Means.")
with right:
    st.markdown("**Vị trí giao dịch · PCA**")
    points = result.sample(n=min(5000, len(result)), random_state=42)[["PC1", "PC2", "cluster", TIME, AMOUNT]].copy()
    points["Nhóm"] = points["cluster"].map(lambda cluster: f"Cụm {cluster}")
    scatter = alt.Chart(points).mark_circle(size=22, opacity=.42).encode(
        x=alt.X("PC1:Q", title=f"PC1 · {pca.explained_variance_ratio_[0]:.1%}"),
        y=alt.Y("PC2:Q", title=f"PC2 · {pca.explained_variance_ratio_[1]:.1%}"), color=color,
        tooltip=["Nhóm:N", alt.Tooltip(f"{TIME}:T", title="Thời gian", format="%d/%m/%Y %H:%M"),
                 alt.Tooltip(f"{AMOUNT}:Q", title="Số tiền (VND)", format=",")]).interactive()
    st.altair_chart(scatter.properties(height=310), width="stretch", alt="Phép chiếu PCA của giao dịch, màu theo cụm.")
    st.caption(f"{number(len(points))} điểm lấy mẫu cố định; hai trục giữ {pca.explained_variance_ratio_.sum():.1%} phương sai. "
               "PCA được học từ train và chỉ dùng để hiển thị. Có thể kéo, thu phóng và rê chuột xem điểm.")

st.markdown("**Đặc điểm các nhóm trong dữ liệu đang xem**")
for column, cluster in zip(st.columns(model.n_clusters), cluster_ids):
    group = result.loc[result["cluster"].eq(cluster)]
    with column, st.container(border=True):
        st.markdown(f"**Cụm {cluster}** · {len(group) / len(result):.1%}")
        if group.empty:
            st.caption("Không có giao dịch trong đợt này.")
            continue
        amount = group[AMOUNT].median()
        st.write(f"{number(len(group))} giao dịch · Trung vị {'—' if pd.isna(amount) else number(amount)} VND")
        st.caption(f"Trung vị số lần người gửi giao dịch trong 7 ngày trước: {group['sender_tx_count_7d'].median():.0f}. "
                   f"Tỷ lệ người nhận mới: {group['is_new_receiver'].mean():.1%}.")
st.caption("Các đặc điểm trên mô tả từng giao dịch theo lịch sử người gửi; không phải phân hạng khách hàng hay mức rủi ro.")

st.subheader("03  Tra cứu giao dịch")
st.caption("Bộ lọc chỉ áp dụng cho bảng và CSV bên dưới. Biểu đồ phía trên thể hiện toàn bộ đợt phân nhóm.")
a, b, c = st.columns([1, 1.2, 1.4])
selected = a.multiselect("Nhóm giao dịch", cluster_ids, default=cluster_ids, format_func=lambda value: f"Cụm {value}", key="cluster_filter")
dates = b.date_input("Khoảng ngày", value=(result[TIME].min().date(), result[TIME].max().date()),
                     min_value=result[TIME].min().date(), max_value=result[TIME].max().date(), format="DD/MM/YYYY", key="date_filter")
query = c.text_input("Tìm mã giao dịch / tài khoản", key="search_filter", placeholder="Nhập toàn bộ hoặc một phần…")
masked = st.checkbox("Ẩn bớt số tài khoản trong bảng và file tải xuống", value=True)
if len(dates) != 2:
    st.info("Chọn đủ ngày bắt đầu và ngày kết thúc.")
    st.stop()
filtered = filter_transactions(result, selected, *dates, query)
table = display_transactions(filtered, masked)
st.caption(f"{number(len(table))} giao dịch phù hợp · Bảng hiển thị tối đa 500 dòng, CSV chứa toàn bộ kết quả đã lọc.")
st.dataframe(table.head(500), hide_index=True, height=380, column_config={
    "Thời gian": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm:ss"),
    "Số tiền (VND)": st.column_config.NumberColumn(format="localized"),
})
st.download_button("Tải kết quả đã lọc · CSV", csv_bytes(table), "ibft_clustering.csv", "text/csv",
                   disabled=table.empty, on_click="ignore")
st.caption("IBFT Explorer · Chạy trên máy của bạn · Kết quả tải xuống không tự lưu vào output/.")
