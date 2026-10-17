#!/usr/bin/env python
# coding=utf-8

import pandas as pd
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text

KEYS = ["Anon Student Id", "KC(SubSkills)", "Questions"]
REQUIRED_COLUMNS = [
    "Anon Student Id", "Problem Name", "Step Name", "KC(SubSkills)",
    "Correct First Attempt", "First Transaction Time",
]
NO_SKILL = {"", "noskill", "nan", "none", "null", "na", "-1"}


def normalize_skills(value):
    """Preserve the full row KC set; shared split code expands it later.

    Unlike ASSIST2017's single-skill rows, KC(SubSkills) already contains
    multi-KC annotations. Do not infer a union from other interactions.
    """
    if pd.isna(value):
        return ""
    skills = []
    for token in str(value).split("~~"):
        token = token.strip()
        if token.lower() in NO_SKILL:
            continue
        skill = replace_text(token)
        if skill not in skills:
            skills.append(skill)
    return "_".join(skills)


def parse_timestamp(value):
    """Use Algebra2005's millisecond conversion with optional fractions."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    for has_fraction in (True, False):
        try:
            return change2timestamp(value, hasf=has_fraction)
        except (ValueError, OverflowError, OSError):
            continue
    return None

def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_table(
        read_file, encoding="utf-8", low_memory=False,
        usecols=REQUIRED_COLUMNS, dtype=str,
    )
    raw_interactions = len(df)
    print(f"original interaction num: {raw_interactions}, user num: {df['Anon Student Id'].nunique()}")
    df["index"] = range(len(df))

    # Validate identity before escaping: missing names must not become 'nan' IDs.
    identity_columns = ["Anon Student Id", "Problem Name", "Step Name"]
    df = df.dropna(subset=identity_columns).copy()
    for column in identity_columns:
        df[column] = df[column].str.strip()
        df = df.loc[df[column] != ""].copy()
    df["Problem Name"] = df["Problem Name"].map(replace_text)
    df["Step Name"] = df["Step Name"].map(replace_text)
    df["Questions"] = df["Problem Name"] + "----" + df["Step Name"]

    df["KC(SubSkills)"] = df["KC(SubSkills)"].map(normalize_skills)
    no_kc = df["KC(SubSkills)"] == ""
    removed_no_kc = int(no_kc.sum())
    df = df.loc[~no_kc].copy()

    df["Correct First Attempt"] = pd.to_numeric(df["Correct First Attempt"], errors="coerce")
    valid_response = df["Correct First Attempt"].isin([0, 1])
    removed_invalid_response = int((~valid_response).sum())
    df = df.loc[valid_response].copy()
    df["Correct First Attempt"] = df["Correct First Attempt"].astype(int)

    df["First Transaction Time"] = df["First Transaction Time"].map(parse_timestamp)
    removed_invalid_timestamp = int(df["First Transaction Time"].isna().sum())
    df = df.dropna(subset=["First Transaction Time"]).copy()
    df["First Transaction Time"] = df["First Transaction Time"].astype("int64")
    print(
        f"removed interactions: {raw_interactions - len(df)}, "
        f"no KC: {removed_no_kc}, invalid response: {removed_invalid_response}, "
        f"invalid timestamp: {removed_invalid_timestamp}"
    )
    if df.empty:
        raise ValueError("Bridge2Algebra2006 has no valid interactions after filtering")

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares, "_")
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    kc_counts = df["KC(SubSkills)"].map(lambda skills: len(skills.split("_")))
    print(f"multi-KC interactions: {int((kc_counts > 1).sum())}, max concepts per interaction: {int(kc_counts.max())}")

    user_inters = []
    for user, group in df.groupby("Anon Student Id", sort=False):
        group = group.sort_values(by=["First Transaction Time", "index"], kind="mergesort")

        seq_problems = group["Questions"].tolist()
        seq_ans = group["Correct First Attempt"].tolist()
        seq_start_time = format_list2str(group["First Transaction Time"].tolist())
        seq_skills = group["KC(SubSkills)"].tolist()
        seq_len = len(seq_ans)
        seq_use_time = ["NA"]

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_start_time)
        
        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, format_list2str(seq_ans), seq_start_time, seq_use_time])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return

