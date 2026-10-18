"""
check_tweet_coverage_final.py
شمارش تطابق یکتا بین user_id متادیتا و شناسه‌های شیت‌های توییت،
با تحمل اختلاف در N رقم آخر (پیش‌فرض ۵)، و محاسبه‌ی اجتماع تطابق‌ها.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------- نرمال‌سازی ----------
def norm_id(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    if s.endswith(".0"):
        s = s[:-2]
    if not s.isdigit():
        return None
    return s


def split_prefix(uid, tolerance):
    if uid is None or len(uid) <= tolerance:
        return None, uid
    return uid[:-tolerance], uid[-tolerance:]


def build_prefix_index(ids, tolerance):
    idx = defaultdict(set)
    for uid in ids:
        pfx, _ = split_prefix(uid, tolerance)
        if pfx is not None:
            idx[pfx].add(uid)
    return idx


# ---------- منطق اصلی ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--users-sheet", required=True)
    ap.add_argument("--tweets-sheets", nargs="+", required=True)
    ap.add_argument("--user-id-col", default="user_id")
    ap.add_argument("--tolerance", type=int, default=5)
    ap.add_argument("--out-csv", default=None,
                    help="ذخیره‌ی وضعیت هر کاربر در متادیتا")
    args = ap.parse_args()

    path = Path(args.xlsx)

    # ---- کاربران ----
    users = pd.read_excel(path, sheet_name=args.users_sheet, dtype=str)
    if args.user_id_col not in users.columns:
        raise SystemExit(f"ستون {args.user_id_col!r} در شیت کاربران نیست.")
    users["__uid"] = users[args.user_id_col].map(norm_id)
    user_ids = set(users["__uid"].dropna())
    print(f"[users] {len(users)} ردیف | {len(user_ids)} شناسه‌ی یکتا (نرمال)")

    user_prefix_index = build_prefix_index(user_ids, args.tolerance)
    print(f"[users] پیشوند یکتا با تحمل {args.tolerance} رقم: {len(user_prefix_index)}")

    # ---- شیت‌های توییت ----
    per_sheet_matches = {}  # name -> set(user_ids)
    for sheet in args.tweets_sheets:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
        candidate_cols = [c for c in df.columns
                          if "id" in c.lower() and "project" not in c.lower()]

        # پیدا کردن بهترین ستون برای این شیت
        best_col, best_matched = None, set()
        for col in candidate_cols:
            ids = set(df[col].map(norm_id).dropna())
            matched = set()
            for uid in ids:
                pfx, _ = split_prefix(uid, args.tolerance)
                if pfx is not None and pfx in user_prefix_index:
                    matched |= user_prefix_index[pfx]
            if len(matched) > len(best_matched):
                best_col, best_matched = col, matched

        if best_col is None:
            print(f"[tweets] شیت {sheet!r}: هیچ ستون نامزدی پیدا نشد.")
            per_sheet_matches[sheet] = set()
            continue

        per_sheet_matches[sheet] = best_matched
        print(f"[tweets] شیت {sheet!r} | ستون={best_col!r} | "
              f"تطابق یکتا با متادیتا={len(best_matched)}")

    # ---- اجتماع ----
    union_matched = set()
    for s in per_sheet_matches.values():
        union_matched |= s

    # فقط کاربرانی که در متادیتا هستند (برای اطمینان)
    union_matched &= user_ids

    total = len(user_ids)
    n_union = len(union_matched)
    pct = 100.0 * n_union / total if total else 0.0

    # ---- چاپ نتیجه ----
    print()
    print("===== نتیجه =====")
    print(f"تعداد اکانت‌های یکتا در متادیتا:              {total}")
    print()
    print(f"--- تطابق یکتا به تفکیک شیت ---")
    for sheet, matched in per_sheet_matches.items():
        print(f"  {sheet!r}: {len(matched)} اکانت")
    print()
    print(f"--- اجتماع تطابق‌های یکتا ---")
    print(f"  |اجتماع| = {n_union}")
    print(f"  درصد پوشش = {pct:.2f}%")

    # ---- تحلیل هم‌پوشانی بین شیت‌ها ----
    if len(per_sheet_matches) >= 2:
        print()
        print("--- هم‌پوشانی بین شیت‌ها ---")
        sheet_names = list(per_sheet_matches.keys())
        for i in range(len(sheet_names)):
            for j in range(i + 1, len(sheet_names)):
                s1 = per_sheet_matches[sheet_names[i]]
                s2 = per_sheet_matches[sheet_names[j]]
                inter = s1 & s2
                only1 = s1 - s2
                only2 = s2 - s1
                print(f"  {sheet_names[i]!r} ∩ {sheet_names[j]!r} = {len(inter)}")
                print(f"  فقط در {sheet_names[i]!r} = {len(only1)}")
                print(f"  فقط در {sheet_names[j]!r} = {len(only2)}")
        print(f"  (بررسی: {sum(len(s) for s in per_sheet_matches.values())} - "
              f"{sum(len(per_sheet_matches[a]) + len(per_sheet_matches[b]) - 2*len(per_sheet_matches[a] & per_sheet_matches[b]) for i, a in enumerate(sheet_names) for b in sheet_names[i+1:])} "
              f"= {n_union})")

    # ---- ذخیره ----
    if args.out_csv:
        out = users.copy()
        out["has_tweet_text"] = out["__uid"].isin(union_matched).astype(int)
        out = out.drop(columns=["__uid"])
        out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
        n_with = int(out["has_tweet_text"].sum())
        print(f"\n[ok] ذخیره شد: {args.out_csv} "
              f"({n_with} ردیف با has_tweet_text=1)")


if __name__ == "__main__":
    main()