from pathlib import Path
import sys

import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "sleep_duration_hours",
    "sleep_quality",
    "study_hours",
    "deep_focus_hours",
    "phone_entertainment_hours",
    "exercise_hours",
    "class_load",
    "focus_score",
    "notes",
]

HOUR_BOUNDS = {
    "sleep_duration_hours": (0, 14),
    "study_hours": (0, 24),
    "deep_focus_hours": (0, 24),
    "phone_entertainment_hours": (0, 24),
    "exercise_hours": (0, 12),
}

VALID_CLASS_LOADS = {"低", "中", "高"}
VALID_SCORES = {1, 2, 3, 4, 5}


def csv_row_numbers(mask: pd.Series) -> list[int]:
    """将 DataFrame 的错误位置转换为 CSV 中的实际行号。"""
    return (mask.to_numpy().nonzero()[0] + 2).tolist()


def validate_learning_log(df: pd.DataFrame) -> list[str]:
    """检查学习行为日志是否满足项目的数据质量规则。"""
    issues = []

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        issues.append(f"缺少必要字段：{', '.join(sorted(missing_columns))}")
        return issues

    dates = pd.to_datetime(df["date"], errors="coerce")

    if dates.isna().any():
        issues.append(f"日期格式无效，CSV 行号：{csv_row_numbers(dates.isna())}")

    if dates.notna().any() and dates.duplicated().any():
        duplicate_dates = dates[dates.duplicated()].dt.strftime("%Y-%m-%d").tolist()
        issues.append(f"存在重复日期记录：{duplicate_dates}")

    for column, (minimum, maximum) in HOUR_BOUNDS.items():
        values = pd.to_numeric(df[column], errors="coerce")

        if values.isna().any():
            issues.append(
                f"{column} 含有非数字或空值，CSV 行号："
                f"{csv_row_numbers(values.isna())}"
            )

        out_of_range = values.notna() & (
            (values < minimum) | (values > maximum)
        )
        if out_of_range.any():
            issues.append(
                f"{column} 必须在 {minimum}–{maximum} 小时之间，"
                f"CSV 行号：{csv_row_numbers(out_of_range)}"
            )

    for column in ["sleep_quality", "focus_score"]:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid_score = values.isna() | ~values.isin(VALID_SCORES)

        if invalid_score.any():
            issues.append(
                f"{column} 只能填写 1–5 分，"
                f"CSV 行号：{csv_row_numbers(invalid_score)}"
            )

    invalid_class_load = ~df["class_load"].isin(VALID_CLASS_LOADS)
    if invalid_class_load.any():
        issues.append(
            "class_load 只能填写“低”“中”或“高”，"
            f"CSV 行号：{csv_row_numbers(invalid_class_load)}"
        )

    study_hours = pd.to_numeric(df["study_hours"], errors="coerce")
    deep_focus_hours = pd.to_numeric(df["deep_focus_hours"], errors="coerce")

    invalid_focus_duration = (
        study_hours.notna()
        & deep_focus_hours.notna()
        & (deep_focus_hours > study_hours)
    )

    if invalid_focus_duration.any():
        issues.append(
            "deep_focus_hours 不能大于 study_hours，"
            f"CSV 行号：{csv_row_numbers(invalid_focus_duration)}"
        )

    return issues


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "sample_learning_log.csv"

    df = pd.read_csv(data_path)
    issues = validate_learning_log(df)

    if issues:
        print("❌ 数据质量检查未通过：")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    print(
        f"✅ 数据质量检查通过："
        f"{len(df)} 条记录、{len(df.columns)} 个字段均符合当前规则。"
    )


if __name__ == "__main__":
    main()