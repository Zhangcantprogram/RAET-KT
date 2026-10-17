"""Canonical NIPS task 3/4 preprocessing using official Level-3 subjects."""
import ast
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _read_csv(path, required):
    frame = pd.read_csv(path, dtype=str, encoding='utf-8-sig', low_memory=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    return frame


def _ids(series):
    numbers = pd.to_numeric(series, errors='coerce')
    valid = numbers.notna() & numbers.ge(0) & numbers.mod(1).eq(0)
    return numbers.where(valid).astype('Int64').astype('string')


def _subject_ids(value):
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        subjects = ast.literal_eval(str(value))
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f'Invalid official SubjectId list: {value!r}') from exc
    if not isinstance(subjects, (list, tuple)):
        raise ValueError(f'SubjectId must be a list: {value!r}')
    for item in subjects:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f'Invalid official SubjectId: {item!r}')
    return [str(item) for item in subjects]


def load_nips_data(primary_data_path, meta_data_dir, task_name='task_3_4'):
    """Join official metadata without multiplying or deduplicating answers."""
    root = Path(meta_data_dir)
    paths = {
        'answers': root / f'answer_metadata_{task_name}.csv',
        'questions': root / f'question_metadata_{task_name}.csv',
        'subjects': root / 'subject_metadata.csv',
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'NIPS task 3/4 requires official KC and answer-time metadata. '
            'Place these files in the metadata directory:\n' + '\n'.join(missing)
        )
    primary = _read_csv(primary_data_path, ['UserId', 'QuestionId', 'AnswerId', 'IsCorrect'])
    primary['original_index'] = range(len(primary))
    for column in ('UserId', 'QuestionId', 'AnswerId'):
        primary[column] = _ids(primary[column])
    primary['IsCorrect'] = pd.to_numeric(primary['IsCorrect'], errors='coerce')
    answers = _read_csv(paths['answers'], ['AnswerId', 'DateAnswered'])
    questions = _read_csv(paths['questions'], ['QuestionId', 'SubjectId'])
    subjects = _read_csv(paths['subjects'], ['SubjectId', 'Level'])
    for frame, key in ((answers, 'AnswerId'), (questions, 'QuestionId'), (subjects, 'SubjectId')):
        frame[key] = _ids(frame[key])
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f'Official metadata must have unique valid {key} values')
    level3 = set(subjects.loc[pd.to_numeric(subjects['Level'], errors='coerce').eq(3), 'SubjectId'])
    if not level3:
        raise ValueError('Official subject metadata contains no Level-3 KCs')
    all_subjects = set(subjects['SubjectId'])

    def kc_set(value):
        ids = set(_subject_ids(value))
        unknown = ids - all_subjects
        if unknown:
            raise ValueError(f'Question metadata references unknown SubjectIds: {sorted(unknown)}')
        return '_'.join(sorted(ids & level3, key=int))

    questions['concepts'] = questions['SubjectId'].map(kc_set)
    # Parse individually to support whole/fractional seconds; UTC avoids
    # dependence on the host timezone. Published timestamps have no zone.
    dates = answers['DateAnswered'].map(lambda value: pd.to_datetime(value, errors='coerce', utc=True))
    answers['answer_timestamp'] = dates.map(
        lambda value: pd.NA if pd.isna(value) else value.value // 1_000_000
    ).astype('Int64')
    merged = primary.merge(answers[['AnswerId', 'answer_timestamp']], on='AnswerId', how='left', validate='many_to_one')
    return merged.merge(questions[['QuestionId', 'concepts']], on='QuestionId', how='left', validate='many_to_one')


def read_data_from_csv(primary_data_path, meta_data_dir, task_name, write_file):
    """Write six lines per student; shared pyKT handles splits and KC expansion."""
    df = load_nips_data(primary_data_path, meta_data_dir, task_name)
    audit = {'dataset': 'nips_task34', 'raw_interactions': len(df)}
    rules = (
        ('removed_invalid_id', df[['UserId', 'QuestionId', 'AnswerId']].notna().all(axis=1)),
        ('removed_invalid_answer', df['IsCorrect'].isin([0, 1])),
        ('removed_unknown_question', df['concepts'].notna()),
        ('removed_no_kc', df['concepts'].fillna('').ne('')),
        ('removed_invalid_timestamp', df['answer_timestamp'].notna() & df['answer_timestamp'].ge(0)),
    )
    keep = pd.Series(True, index=df.index)
    for label, valid in rules:
        valid = valid.fillna(False)
        audit[label] = int((keep & ~valid).sum())
        keep &= valid
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError('No valid NIPS task 3/4 interactions remain after filtering')
    df['IsCorrect'] = df['IsCorrect'].astype(int)
    df['answer_timestamp'] = df['answer_timestamp'].astype('int64')
    counts = df['concepts'].str.count('_') + 1
    question_kcs = df.drop_duplicates('QuestionId')
    question_counts = question_kcs['concepts'].str.count('_') + 1
    concepts = {cid for value in question_kcs['concepts'] for cid in value.split('_')}
    incorrect = df['IsCorrect'].eq(0)
    multi = counts.gt(1)
    audit.update(
        processed_students=int(df['UserId'].nunique()),
        valid_interactions=len(df), num_questions=len(question_kcs), num_concepts=len(concepts),
        max_concepts=int(counts.max()), multi_kc_questions=int(question_counts.gt(1).sum()),
        multi_kc_question_ratio=float(question_counts.gt(1).mean()),
        interaction_k1=int(counts.eq(1).sum()), interaction_k2=int(counts.eq(2).sum()),
        interaction_k3plus=int(counts.ge(3).sum()), correct_interactions=int((~incorrect).sum()),
        incorrect_interactions=int(incorrect.sum()), incorrect_multi_kc=int((incorrect & multi).sum()),
        multi_kc_interaction_ratio=float(multi.mean()),
        incorrect_multi_ratio_all=float((incorrect & multi).mean()),
        incorrect_multi_ratio_incorrect=float((incorrect & multi).sum() / incorrect.sum()) if incorrect.any() else 0.0,
        question_kc_source='official question SubjectId intersected with official Level-3 subjects',
        ordering='DateAnswered (UTC milliseconds) -> original_index',
    )
    output = Path(write_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        groups = df.groupby('UserId', sort=False)
        for uid, group in tqdm(groups, total=audit['processed_students'], desc='Preprocessing NIPS task 3/4', unit='student'):
            group = group.sort_values(['answer_timestamp', 'original_index'], kind='stable')
            handle.write(f'{uid},{len(group)}\n')
            for column in ('QuestionId', 'concepts', 'IsCorrect', 'answer_timestamp'):
                handle.write(','.join(group[column].astype(str)) + '\n')
            handle.write('NA\n')
    temporary.replace(output)
    with (output.parent / 'preprocess_audit.json').open('w', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
