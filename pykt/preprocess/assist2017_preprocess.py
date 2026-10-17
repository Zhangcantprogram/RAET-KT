import pandas as pd
from .utils import sta_infos, write_txt, format_list2str, replace_text

keys = ["studentId", "skill", "problemId"]
NO_SKILL = "noskill"


def build_question_to_skills(df):
    """Build a deterministic problem -> KC-set mapping.

    The shared pyKT preprocessing pipeline represents a multi-KC item by
    joining its concepts with ``_``.  ``extend_multi_concepts`` later expands
    that value and creates the ``is_repeat`` markers consumed by RAPLKT.

    ASSISTments 2017 stores only one skill on each interaction row, although
    the same problem can be tagged with different skills on other rows.  We
    therefore treat the union of its real tags as item metadata and attach the
    complete set to every interaction of the problem.  ``noskill`` is not a
    KC; questions with no real tag are omitted by ``read_data_from_csv``.
    """
    question_to_skills = {}

    for question, skill in zip(df["problemId"], df["skill"]):
        skill = str(skill).strip()
        if skill.lower() == NO_SKILL:
            continue

        # ``_`` is the multi-concept protocol delimiter and ``,`` is the
        # sequence delimiter, so protect both inside raw skill names exactly
        # as the Algebra2005 reader does.
        skill = replace_text(skill)
        skills = question_to_skills.setdefault(question, [])
        if skill not in skills:
            skills.append(skill)

    return question_to_skills


def read_data_from_csv(read_file, write_file):
    df = pd.read_csv(read_file, encoding='utf-8', low_memory=False)

    stares = []
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(
        df, keys, stares, split_str="~~"
    )
    print(
        f"original interaction num: {df.shape[0]}, user num: {df['studentId'].nunique()}, question num: {df['problemId'].nunique()}, "
        f"concept num: {df['skill'].nunique()}, avg(ins) per s:{avgins}, avg(c) per q:{avgcq}, na:{na}")

    df["index"] = range(len(df))

    df = df.dropna(subset=["studentId", "problemId", "correct", "skill", "startTime"])
    df = df[df['correct'].isin([0, 1])]  
    df.loc[:, 'timeTaken'] = df['timeTaken'].apply(lambda x: round(x * 1000))

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(
        df, keys, stares, split_str="~~"
    )
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df2 = df[["index", "studentId", "problemId", "skill", "correct", "timeTaken", "startTime"]]
    question_to_skills = build_question_to_skills(df2)
    df2 = df2.copy()
    has_real_concept = df2["problemId"].isin(question_to_skills)
    removed_interactions = int((~has_real_concept).sum())
    removed_questions = int(
        df2.loc[~has_real_concept, "problemId"].nunique()
    )
    df2 = df2.loc[has_real_concept].copy()
    df2.loc[:, "skill"] = df2["problemId"].map(
        lambda question: "_".join(question_to_skills[question])
    )

    concept_counts = [len(skills) for skills in question_to_skills.values()]
    multi_concept_questions = sum(count > 1 for count in concept_counts)
    max_concepts = max(concept_counts, default=0)
    print(
        "question-level KC mapping: "
        f"multi-concept questions: {multi_concept_questions}, "
        f"max concepts per question: {max_concepts}, "
        f"removed noskill-only questions: {removed_questions}, "
        f"removed interactions: {removed_interactions}"
    )

    ui_df = df2.groupby('studentId', sort=False)

    user_inter = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]  
        tmp_inter.loc[:, 'startTime'] = tmp_inter.loc[:, 'startTime'].apply(lambda t: int(t) * 1000)
        tmp_inter = tmp_inter.sort_values(by=['startTime', 'index'])

        tmp_inter['startTime'] = tmp_inter['startTime']

        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['problemId'].tolist()
        seq_skills = tmp_inter['skill'].tolist()
        seq_ans = tmp_inter['correct'].tolist()
        seq_submit_time = tmp_inter['startTime'].tolist()
        seq_response_cost = tmp_inter['timeTaken'].tolist()

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_submit_time) == len(seq_response_cost)

        user_inter.append(
            [[str(user), str(seq_len)], format_list2str(seq_problems), seq_skills, format_list2str(seq_ans), format_list2str(seq_submit_time), format_list2str(seq_response_cost)])

    write_txt(write_file, user_inter)


