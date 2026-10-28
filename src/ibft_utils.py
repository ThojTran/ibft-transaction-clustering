"""Đọc giao dịch và suy luận bằng đúng đặc trưng của notebook 01."""

from pathlib import Path

import numpy as np
import pandas as pd

TIME = "transaction_datetime"
AMOUNT = "transactionamount"
SENDER = "accountsourcenumber"
RECEIVER = "accountdestnumber"
REFERENCE = "referencenumber"
FEATURES = [
    "log_amount", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "log_sender_recency_days", "log_sender_tx_count_7d", "log_pair_tx_count_7d",
    "receiver_share_7d", "is_first_sender_tx", "is_new_receiver",
]
X_COLUMNS = [f"x_{name}" for name in FEATURES]
TEMPLATE_COLUMNS = [REFERENCE, TIME, SENDER, RECEIVER, AMOUNT, "responsecode"]


def read_transactions(source, filename, source_key=None):
    """Excel: sheet IBFT hoặc một sheet duy nhất. CSV: UTF-8, dấu phẩy."""
    identifiers = {SENDER: "string", RECEIVER: "string", REFERENCE: "string"}
    if Path(filename).suffix.lower() == ".csv":
        data = pd.read_csv(source, dtype=identifiers, encoding="utf-8-sig")
    else:
        with pd.ExcelFile(source) as book:
            if "IBFT" in book.sheet_names:
                sheet = "IBFT"
            elif len(book.sheet_names) == 1:
                sheet = book.sheet_names[0]
            else:
                raise ValueError("File Excel nhiều sheet cần có sheet tên IBFT.")
            data = pd.read_excel(book, sheet_name=sheet, dtype=identifiers)

    required = {SENDER, RECEIVER, AMOUNT, REFERENCE, "responsecode"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Thiếu cột: " + ", ".join(sorted(missing)))
    for column in identifiers:
        data[column] = data[column].str.strip().replace("", pd.NA)
    if TIME in data:
        # Giờ địa phương Việt Nam, ISO: YYYY-MM-DD HH:MM:SS.
        try:
            parsed = pd.to_datetime(data[TIME], format="ISO8601", errors="coerce")
            if parsed.dt.tz is not None:
                parsed = parsed.dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
            data[TIME] = parsed
        except (ValueError, AttributeError) as exc:
            raise ValueError("Thời gian cần cùng múi giờ và dạng YYYY-MM-DD HH:MM:SS.") from exc
    elif {"short_filename", "timelocaltranx"}.issubset(data.columns):
        date_text = data["short_filename"].astype("string").str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
        clock = pd.to_numeric(data["timelocaltranx"], errors="coerce")
        clock = clock.where(clock.mod(1).eq(0)).astype("Int64").astype("string").str.zfill(6)
        data[TIME] = pd.to_datetime(date_text + " " + clock, format="%Y-%m-%d %H%M%S", errors="coerce")
    else:
        raise ValueError("Cần transaction_datetime hoặc cả short_filename và timelocaltranx.")
    data[AMOUNT] = pd.to_numeric(data[AMOUNT], errors="coerce")
    data["responsecode"] = pd.to_numeric(data["responsecode"], errors="coerce")
    data["transaction_id"] = [f"{source_key or Path(filename).stem}:{row}" for row in range(2, len(data) + 2)]
    return data[["transaction_id", *TEMPLATE_COLUMNS]]


def clean_transactions(data, require_reference=False):
    """Giữ giao dịch thành công; khử trùng trước khi tính lịch sử."""
    valid = (
        data["responsecode"].eq(0) & data[TIME].notna()
        & data[SENDER].notna() & data[RECEIVER].notna()
        & (data[AMOUNT].isna() | (data[AMOUNT].gt(0) & np.isfinite(data[AMOUNT])))
    )
    if require_reference:
        valid &= data[REFERENCE].notna()
    clean = data.loc[valid].copy()
    if require_reference:
        fields = [TIME, SENDER, RECEIVER, AMOUNT]
        if clean.groupby(REFERENCE)[fields].nunique(dropna=False).gt(1).any().any():
            raise ValueError("Một mã giao dịch có nhiều nội dung khác nhau. Hãy kiểm tra các dòng trùng mã.")
    duplicate = clean[REFERENCE].notna() & clean[REFERENCE].duplicated()
    stats = {"received": len(data), "invalid": int((~valid).sum()), "duplicates": int(duplicate.sum())}
    clean = clean.loc[~duplicate].sort_values([TIME, "transaction_id"]).reset_index(drop=True)
    return clean, stats


def build_features(data):
    """Cửa sổ [t - 7 ngày, t); giao dịch cùng thời điểm không thấy nhau."""
    data = data.copy()
    for keys, count_name in [([SENDER], "sender_tx_count_7d"), ([SENDER, RECEIVER], "pair_tx_count_7d")]:
        history = (data.groupby([*keys, TIME]).size().reset_index(name="at_time")
                   .sort_values([*keys, TIME]).reset_index(drop=True))
        if len(keys) == 1:
            history["sender_previous_time"] = history.groupby(keys)[TIME].shift(1)
            extras = ["sender_previous_time"]
        else:
            history["is_new_receiver"] = history.groupby(keys).cumcount().eq(0).astype(int)
            extras = ["is_new_receiver"]
        counts = (history.groupby(keys, sort=False).rolling("7D", on=TIME, closed="left")["at_time"]
                  .sum().rename(count_name).reset_index())
        history = history.merge(counts, on=[*keys, TIME], validate="one_to_one")
        history[count_name] = history[count_name].fillna(0).astype(int)
        data = data.merge(history[[*keys, TIME, count_name, *extras]], on=[*keys, TIME], validate="many_to_one")

    recency = (data[TIME] - data["sender_previous_time"]).dt.total_seconds() / 86400
    data["is_first_sender_tx"] = data["sender_previous_time"].isna().astype(int)
    data["receiver_share_7d"] = data["pair_tx_count_7d"].div(data["sender_tx_count_7d"].replace(0, np.nan)).fillna(0)
    seconds = data[TIME].dt.hour * 3600 + data[TIME].dt.minute * 60 + data[TIME].dt.second
    data["hour_sin"] = np.sin(2 * np.pi * seconds / 86400)
    data["hour_cos"] = np.cos(2 * np.pi * seconds / 86400)
    data["weekday_sin"] = np.sin(2 * np.pi * data[TIME].dt.dayofweek / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data[TIME].dt.dayofweek / 7)
    data["log_amount"] = np.log1p(data[AMOUNT])
    data["log_sender_recency_days"] = np.log1p(recency)
    data["log_sender_tx_count_7d"] = np.log1p(data["sender_tx_count_7d"])
    data["log_pair_tx_count_7d"] = np.log1p(data["pair_tx_count_7d"])
    return data.sort_values([TIME, "transaction_id"]).reset_index(drop=True)


def predict_frame(data, matrix, model, pca):
    result = data.copy()
    matrix = np.asarray(matrix, dtype=np.float64)
    result["cluster"] = model.predict(matrix)
    result[["PC1", "PC2"]] = pca.transform(matrix)
    return result


def predict_upload(raw, history, preprocessor, model, pca):
    target, stats = clean_transactions(raw, require_reference=True)
    if target.empty:
        raise ValueError("Không có giao dịch hợp lệ. Kiểm tra mã giao dịch, thời gian, tài khoản, số tiền và responsecode = 0.")
    history = history.copy()
    if not history.empty:
        # Cùng giao dịch đã có trong lịch sử chỉ được tính một lần.
        overlap = target.merge(history, on=REFERENCE, suffixes=("_new", "_old"))
        for column in [TIME, SENDER, RECEIVER, AMOUNT]:
            left, right = overlap[f"{column}_new"], overlap[f"{column}_old"]
            equal = left.eq(right) | (left.isna() & right.isna())
            if not equal.fillna(False).all():
                raise ValueError("Mã giao dịch trùng lịch sử nhưng nội dung khác. Hãy kiểm tra file hoặc tắt dùng lịch sử có sẵn.")
        history = history.loc[~history[REFERENCE].isin(target[REFERENCE])]
    combined = target.assign(_target=True)
    if not history.empty:
        combined = pd.concat([history.assign(_target=False), combined], ignore_index=True)
    # Chỉ lịch sử trước giao dịch được dùng, kể cả khi file có dữ liệu tương lai.
    combined = combined.loc[combined[TIME].le(target[TIME].max())]
    cutoff = combined[TIME].min() + pd.Timedelta(days=7)
    features = build_features(combined)
    ready = features.loc[features["_target"] & features[TIME].ge(cutoff)].copy()
    stats["warmup"] = len(target) - len(ready)
    if ready.empty:
        raise ValueError("Chưa đủ 7 ngày lịch sử nền. Hãy thêm dữ liệu lịch sử vào file hoặc bật dùng lịch sử trong data/.")
    matrix = preprocessor.transform(ready[FEATURES])
    stats["scored"] = len(ready)
    stats["missing_amount"] = int(ready[AMOUNT].isna().sum())
    return predict_frame(ready, matrix, model, pca), stats


def filter_transactions(data, clusters, start, end, query=""):
    selected = data["cluster"].isin(clusters)
    selected &= data[TIME].ge(pd.Timestamp(start)) & data[TIME].lt(pd.Timestamp(end) + pd.Timedelta(days=1))
    if query.strip():
        match = pd.Series(False, index=data.index)
        for column in [REFERENCE, SENDER, RECEIVER]:
            match |= data[column].astype("string").str.contains(query.strip(), case=False, regex=False, na=False)
        selected &= match
    return data.loc[selected].copy()


def display_transactions(data, masked=True):
    columns = [REFERENCE, TIME, SENDER, RECEIVER, AMOUNT, "cluster"]
    table = data[columns].copy()
    if masked:
        for column in [SENDER, RECEIVER]:
            table[column] = table[column].astype("string").map(
                lambda value: "••••" + value[-4:] if isinstance(value, str) and len(value) > 4 else "••••"
            )
    return table.rename(columns={REFERENCE: "Mã giao dịch", TIME: "Thời gian", SENDER: "Tài khoản gửi",
                                 RECEIVER: "Tài khoản nhận", AMOUNT: "Số tiền (VND)", "cluster": "Cụm"})


def csv_bytes(table):
    """UTF-8 BOM cho Excel; chặn nội dung văn bản bị hiểu thành công thức."""
    safe = table.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value if isinstance(value, str) and
            (value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n"))) else value
        )
    return safe.to_csv(index=False).encode("utf-8-sig")
