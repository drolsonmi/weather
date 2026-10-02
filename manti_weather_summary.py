#!/usr/bin/env python3
"""
manti_weather_summary.py

Summarize normal and record temperature/precipitation data for MANTI, UT
for a given date, plus the day before and the day after.

Data files (tab-separated, as downloaded) expected in a "data" folder next
to this script, unless --data-dir is given:
    manti_avgT_normal.tsv     - Normal average temperature (deg F)
    manti_maxT_normal.tsv     - Normal high temperature (deg F)
    manti_maxT_record.tsv     - Record high temperature (deg F) + year
    manti_minT_normal.tsv     - Normal low temperature (deg F)
    manti_minT_record.tsv     - Record low temperature (deg F) + year
    manti_precip_normal.tsv   - Normal precipitation (inches)
    manti_precip_record.tsv   - Record precipitation (inches)
                                 NOTE: the source file for this one only
                                 contains the record amount for each day,
                                 not the year it occurred, so "Record
                                 Precip Year" is always reported as N/A.

Usage:
    python3 manti_weather_summary.py --date 07-04
    python3 manti_weather_summary.py --date 2026-12-31
    python3 manti_weather_summary.py --date 2/29 --year 2024
    python3 manti_weather_summary.py --date 07-04 --csv out.csv
"""

import argparse
import csv
import datetime
import os
import sys

import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

DATA_FILES = {
    "avgT_normal": "MantiRecords/manti_avgT_normal.tsv",
    "maxT_normal": "MantiRecords/manti_maxT_normal.tsv",
    "maxT_record": "MantiRecords/manti_maxT_record.tsv",
    "minT_normal": "MantiRecords/manti_minT_normal.tsv",
    "minT_record": "MantiRecords/manti_minT_record.tsv",
    "precip_normal": "MantiRecords/manti_precip_normal.tsv",
    "precip_record": "MantiRecords/manti_precip_record.tsv",
}


def _read_rows(path):
    """Read a csv file and return a list of comma-split rows (raw strings)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        return [row for row in reader if row]


def _find_header_index(rows):
    """Find the index of the row whose first cell is 'Day'."""
    for i, row in enumerate(rows):
        if row and row[0].strip() == "Day":
            return i
    raise ValueError("Could not find a 'Day' header row in file")


def _to_number(value):
    value = value.strip()
    if value in ("", "-", "M", "N/A"):
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return None


def _load_simple_table(path):
    """
    Load a 'Day, Jan, Feb, ..., Dec' style table (normal-value files).
    Returns dict[(month_abbr, day)] -> value (float/int or None)
    """
    rows = _read_rows(path)
    header_i = _find_header_index(rows)
    data_rows = rows[header_i + 1:]

    table = {}
    for row in data_rows:
        if not row or not row[0].strip().isdigit():
            continue
        day = int(row[0].strip())
        for month_i, month in enumerate(MONTHS):
            col = 1 + month_i
            if col < len(row):
                table[(month, day)] = _to_number(row[col])
            else:
                table[(month, day)] = None
    return table


def _load_value_year_table(path):
    """
    Load a 'Day, Jan, Jan-Year, Feb, Feb-Year, ...' style table
    (record files). If a file is missing the '-Year' columns in the
    actual data rows (as manti_precip_record.tsv does), the year is
    simply recorded as None for every entry.
    Returns dict[(month_abbr, day)] -> (value, year_or_None)
    """
    rows = _read_rows(path)
    header_i = _find_header_index(rows)
    data_rows = rows[header_i + 1:]

    # Does each data row actually carry 12 value/year pairs (25 cols
    # including Day), or only 12 plain values (13 cols including Day)?
    sample_len = 0
    for row in data_rows:
        if row and row[0].strip().isdigit():
            sample_len = len(row)
            break
    has_years = sample_len >= 25

    table = {}
    for row in data_rows:
        if not row or not row[0].strip().isdigit():
            continue
        day = int(row[0].strip())
        for month_i, month in enumerate(MONTHS):
            if has_years:
                vcol = 1 + month_i * 2
                ycol = vcol + 1
                value = _to_number(row[vcol]) if vcol < len(row) else None
                year_raw = row[ycol].strip() if ycol < len(row) else ""
                year = int(year_raw) if year_raw.isdigit() else None
            else:
                vcol = 1 + month_i
                value = _to_number(row[vcol]) if vcol < len(row) else None
                year = None
            table[(month, day)] = (value, year)
    return table


def load_all_data(data_dir):
    data = {}
    for key, fname in DATA_FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing expected data file: {path}")
        if key in ("maxT_record", "minT_record", "precip_record"):
            data[key] = _load_value_year_table(path)
        else:
            data[key] = _load_simple_table(path)
    return data


def fmt(value, unit=""):
    if value is None:
        return "N/A"
    return f"{value}{unit}"


def build_row(data, date_obj):
    month = MONTHS[date_obj.month - 1]
    day = date_obj.day
    key = (month, day)

    avg_t = data["avgT_normal"].get(key)
    max_t_n = data["maxT_normal"].get(key)
    min_t_n = data["minT_normal"].get(key)
    precip_n = data["precip_normal"].get(key)

    max_t_r, max_t_r_year = data["maxT_record"].get(key, (None, None))
    min_t_r, min_t_r_year = data["minT_record"].get(key, (None, None))
    precip_r, precip_r_year = data["precip_record"].get(key, (None, None))

    return {
        "Date": date_obj.strftime("%Y-%m-%d (%a)"),
        "Normal Avg Temp (F)": fmt(avg_t),
        "Normal High (F)": fmt(max_t_n),
        "Record High (F)": fmt(max_t_r),
        "Record High Year": fmt(max_t_r_year),
        "Normal Low (F)": fmt(min_t_n),
        "Record Low (F)": fmt(min_t_r),
        "Record Low Year": fmt(min_t_r_year),
        "Normal Precip (in)": fmt(precip_n),
        "Record Precip (in)": fmt(precip_r),
        "Record Precip Year": fmt(precip_r_year),
    }


COLUMNS = [
    "Date",
    "Normal Avg Temp (F)",
    "Normal High (F)",
    "Record High (F)",
    "Record High Year",
    "Normal Low (F)",
    "Record Low (F)",
    "Record Low Year",
    "Normal Precip (in)",
    "Record Precip (in)",
    "Record Precip Year",
]


def build_dataframe(rows):
    """
    Build a DataFrame with the dates as columns and the weather
    variables as rows.
    """
    df = pd.DataFrame(rows, columns=COLUMNS).set_index("Date").T
    df.index.name = None
    df.columns.name = None
    return df


# Rows that start a new section; a separator line is drawn above each.
SECTION_STARTS = {
    "Normal Avg Temp (F)",
    "Normal High (F)",
    "Normal Low (F)",
    "Normal Precip (in)",
}


def print_table(df):
    """
    Print the transposed table (dates as columns, variables as rows),
    with a separator line above each section-starting row.
    """
    label_w = max(len(str(i)) for i in df.index)
    col_ws = [max(len(str(c)), *(len(str(v)) for v in df[c]))
              for c in df.columns]

    sep = "-+-".join(["-" * label_w] + ["-" * w for w in col_ws])
    header = " | ".join([" " * label_w] +
                        [str(c).center(w) for c, w in zip(df.columns, col_ws)])
    print(header)

    for label, row in df.iterrows():
        if label in SECTION_STARTS:
            print(sep)
        cells = [str(v).rjust(w) for v, w in zip(row, col_ws)]
        print(" | ".join([str(label).ljust(label_w)] + cells))


def parse_date_arg(date_str, year_arg):
    """
    Accepts:
      MM-DD, MM/DD           (year defaults to --year or current year)
      YYYY-MM-DD, YYYY/MM/DD
    """
    date_str = date_str.strip()
    for sep in ("-", "/"):
        parts = date_str.split(sep)
        if len(parts) == 3:
            y, m, d = (int(p) for p in parts)
            return datetime.date(y, m, d)
        if len(parts) == 2:
            m, d = (int(p) for p in parts)
            y = year_arg if year_arg else datetime.date.today().year
            return datetime.date(y, m, d)
    raise ValueError(f"Could not parse date: {date_str!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize normal/record weather for MANTI, UT for a "
                    "given date, plus the day before and after.")
    parser.add_argument("--date", required=True,
                         help="Date as MM-DD, MM/DD, or YYYY-MM-DD. "
                              "If no year is given, --year or the current "
                              "year is used (only affects weekday display "
                              "and Feb 29 handling).")
    parser.add_argument("--year", type=int, default=None,
                         help="Year to use when --date omits one "
                              "(default: current year).")
    parser.add_argument("--data-dir", default=None,
                         help="Folder containing the .tsv data files "
                              "(default: a 'data' folder next to this "
                              "script).")
    parser.add_argument("--csv", default=None,
                         help="Optional path to also write the table as CSV.")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data")

    try:
        target_date = parse_date_arg(args.date, args.year)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        data = load_all_data(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    day_before = target_date - datetime.timedelta(days=1)
    day_after = target_date + datetime.timedelta(days=1)

    rows = [build_row(data, d) for d in (day_before, target_date, day_after)]

    print(f"Weather summary for MANTI, UT — {target_date.strftime('%B %d')} "
          f"(and the day before/after)\n")
    df = build_dataframe(rows)
    print_table(df)

    if args.csv:
        df.to_csv(args.csv)
        print(f"\nSaved CSV to: {args.csv}")


if __name__ == "__main__":
    main()


    
# #!/usr/bin/env python3
# """
# manti_weather_summary.py

# Summarize normal and record temperature/precipitation data for MANTI, UT
# for a given date, plus the day before and the day after.

# Data files (tab-separated, as downloaded) expected in a "data" folder next
# to this script, unless --data-dir is given:
#     manti_avgT_normal.tsv     - Normal average temperature (deg F)
#     manti_maxT_normal.tsv     - Normal high temperature (deg F)
#     manti_maxT_record.tsv     - Record high temperature (deg F) + year
#     manti_minT_normal.tsv     - Normal low temperature (deg F)
#     manti_minT_record.tsv     - Record low temperature (deg F) + year
#     manti_precip_normal.tsv   - Normal precipitation (inches)
#     manti_precip_record.tsv   - Record precipitation (inches)
#                                  NOTE: the source file for this one only
#                                  contains the record amount for each day,
#                                  not the year it occurred, so "Record
#                                  Precip Year" is always reported as N/A.

# Usage:
#     python3 manti_weather_summary.py --date 07-04
#     python3 manti_weather_summary.py --date 2026-12-31
#     python3 manti_weather_summary.py --date 2/29 --year 2024
#     python3 manti_weather_summary.py --date 07-04 --csv out.csv
# """

# import argparse
# import csv
# import datetime
# import os
# import sys

# MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
#           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# DATA_FILES = {
#     "avgT_normal": "MantiRecords/manti_avgT_normal.tsv",
#     "maxT_normal": "MantiRecords/manti_maxT_normal.tsv",
#     "maxT_record": "MantiRecords/manti_maxT_record.tsv",
#     "minT_normal": "MantiRecords/manti_minT_normal.tsv",
#     "minT_record": "MantiRecords/manti_minT_record.tsv",
#     "precip_normal": "MantiRecords/manti_precip_normal.tsv",
#     "precip_record": "MantiRecords/manti_precip_record.tsv",
# }


# def _read_rows(path):
#     """Read a csv file and return a list of comma-split rows (raw strings)."""
#     with open(path, newline="", encoding="utf-8") as f:
#         reader = csv.reader(f, delimiter="\t")
#         return [row for row in reader if row]


# def _find_header_index(rows):
#     """Find the index of the row whose first cell is 'Day'."""
#     for i, row in enumerate(rows):
#         if row and row[0].strip() == "Day":
#             return i
#     raise ValueError("Could not find a 'Day' header row in file")


# def _to_number(value):
#     value = value.strip()
#     if value in ("", "-", "M", "N/A"):
#         return None
#     try:
#         if "." in value:
#             return float(value)
#         return int(value)
#     except ValueError:
#         return None


# def _load_simple_table(path):
#     """
#     Load a 'Day, Jan, Feb, ..., Dec' style table (normal-value files).
#     Returns dict[(month_abbr, day)] -> value (float/int or None)
#     """
#     rows = _read_rows(path)
#     header_i = _find_header_index(rows)
#     data_rows = rows[header_i + 1:]

#     table = {}
#     for row in data_rows:
#         if not row or not row[0].strip().isdigit():
#             continue
#         day = int(row[0].strip())
#         for month_i, month in enumerate(MONTHS):
#             col = 1 + month_i
#             if col < len(row):
#                 table[(month, day)] = _to_number(row[col])
#             else:
#                 table[(month, day)] = None
#     return table


# def _load_value_year_table(path):
#     """
#     Load a 'Day, Jan, Jan-Year, Feb, Feb-Year, ...' style table
#     (record files). If a file is missing the '-Year' columns in the
#     actual data rows (as manti_precip_record.tsv does), the year is
#     simply recorded as None for every entry.
#     Returns dict[(month_abbr, day)] -> (value, year_or_None)
#     """
#     rows = _read_rows(path)
#     header_i = _find_header_index(rows)
#     data_rows = rows[header_i + 1:]

#     # Does each data row actually carry 12 value/year pairs (25 cols
#     # including Day), or only 12 plain values (13 cols including Day)?
#     sample_len = 0
#     for row in data_rows:
#         if row and row[0].strip().isdigit():
#             sample_len = len(row)
#             break
#     has_years = sample_len >= 25

#     table = {}
#     for row in data_rows:
#         if not row or not row[0].strip().isdigit():
#             continue
#         day = int(row[0].strip())
#         for month_i, month in enumerate(MONTHS):
#             if has_years:
#                 vcol = 1 + month_i * 2
#                 ycol = vcol + 1
#                 value = _to_number(row[vcol]) if vcol < len(row) else None
#                 year_raw = row[ycol].strip() if ycol < len(row) else ""
#                 year = int(year_raw) if year_raw.isdigit() else None
#             else:
#                 vcol = 1 + month_i
#                 value = _to_number(row[vcol]) if vcol < len(row) else None
#                 year = None
#             table[(month, day)] = (value, year)
#     return table


# def load_all_data(data_dir):
#     data = {}
#     for key, fname in DATA_FILES.items():
#         path = os.path.join(data_dir, fname)
#         if not os.path.isfile(path):
#             raise FileNotFoundError(f"Missing expected data file: {path}")
#         if key in ("maxT_record", "minT_record", "precip_record"):
#             data[key] = _load_value_year_table(path)
#         else:
#             data[key] = _load_simple_table(path)
#     return data


# def fmt(value, unit=""):
#     if value is None:
#         return "N/A"
#     return f"{value}{unit}"


# def build_row(data, date_obj):
#     month = MONTHS[date_obj.month - 1]
#     day = date_obj.day
#     key = (month, day)

#     avg_t = data["avgT_normal"].get(key)
#     max_t_n = data["maxT_normal"].get(key)
#     min_t_n = data["minT_normal"].get(key)
#     precip_n = data["precip_normal"].get(key)

#     max_t_r, max_t_r_year = data["maxT_record"].get(key, (None, None))
#     min_t_r, min_t_r_year = data["minT_record"].get(key, (None, None))
#     precip_r, precip_r_year = data["precip_record"].get(key, (None, None))

#     return {
#         "Date": date_obj.strftime("%Y-%m-%d (%a)"),
#         "Normal Avg Temp (F)": fmt(avg_t),
#         "Normal High (F)": fmt(max_t_n),
#         "Record High (F)": fmt(max_t_r),
#         "Record High Year": fmt(max_t_r_year),
#         "Normal Low (F)": fmt(min_t_n),
#         "Record Low (F)": fmt(min_t_r),
#         "Record Low Year": fmt(min_t_r_year),
#         "Normal Precip (in)": fmt(precip_n),
#         "Record Precip (in)": fmt(precip_r),
#         "Record Precip Year": fmt(precip_r_year),
#     }


# COLUMNS = [
#     "Date",
#     "Normal Avg Temp (F)",
#     "Normal High (F)",
#     "Record High (F)",
#     "Record High Year",
#     "Normal Low (F)",
#     "Record Low (F)",
#     "Record Low Year",
#     "Normal Precip (in)",
#     "Record Precip (in)",
#     "Record Precip Year",
# ]


# def print_table(rows):
#     widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in COLUMNS}
#     sep = "-+-".join("-" * widths[c] for c in COLUMNS)
#     header = " | ".join(c.ljust(widths[c]) for c in COLUMNS)
#     print(header)
#     print(sep)
#     for r in rows:
#         print(" | ".join(r[c].ljust(widths[c]) for c in COLUMNS))


# def write_csv(rows, path):
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=COLUMNS)
#         writer.writeheader()
#         for r in rows:
#             writer.writerow(r)


# def parse_date_arg(date_str, year_arg):
#     """
#     Accepts:
#       MM-DD, MM/DD           (year defaults to --year or current year)
#       YYYY-MM-DD, YYYY/MM/DD
#     """
#     date_str = date_str.strip()
#     for sep in ("-", "/"):
#         parts = date_str.split(sep)
#         if len(parts) == 3:
#             y, m, d = (int(p) for p in parts)
#             return datetime.date(y, m, d)
#         if len(parts) == 2:
#             m, d = (int(p) for p in parts)
#             y = year_arg if year_arg else datetime.date.today().year
#             return datetime.date(y, m, d)
#     raise ValueError(f"Could not parse date: {date_str!r}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Summarize normal/record weather for MANTI, UT for a "
#                     "given date, plus the day before and after.")
#     parser.add_argument("--date", required=True,
#                          help="Date as MM-DD, MM/DD, or YYYY-MM-DD. "
#                               "If no year is given, --year or the current "
#                               "year is used (only affects weekday display "
#                               "and Feb 29 handling).")
#     parser.add_argument("--year", type=int, default=None,
#                          help="Year to use when --date omits one "
#                               "(default: current year).")
#     parser.add_argument("--data-dir", default=None,
#                          help="Folder containing the .tsv data files "
#                               "(default: a 'data' folder next to this "
#                               "script).")
#     parser.add_argument("--csv", default=None,
#                          help="Optional path to also write the table as CSV.")
#     args = parser.parse_args()

#     data_dir = args.data_dir or os.path.join(
#         os.path.dirname(os.path.abspath(__file__)), "data")

#     try:
#         target_date = parse_date_arg(args.date, args.year)
#     except ValueError as e:
#         print(f"Error: {e}", file=sys.stderr)
#         sys.exit(1)

#     try:
#         data = load_all_data(data_dir)
#     except FileNotFoundError as e:
#         print(f"Error: {e}", file=sys.stderr)
#         sys.exit(1)

#     day_before = target_date - datetime.timedelta(days=1)
#     day_after = target_date + datetime.timedelta(days=1)

#     rows = [build_row(data, d) for d in (day_before, target_date, day_after)]

#     print(f"Weather summary for MANTI, UT — {target_date.strftime('%B %d')} "
#           f"(and the day before/after)\n")
#     print_table(rows)

#     if args.csv:
#         write_csv(rows, args.csv)
#         print(f"\nSaved CSV to: {args.csv}")


# if __name__ == "__main__":
#     main()
