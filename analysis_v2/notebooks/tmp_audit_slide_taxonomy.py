import json
from pathlib import Path
from collections import Counter

root = Path('/Users/maxchalekson/Projects/NICO-Research/gemini_data_analysis/outputs')

exp = {
    'broader scientific significance': {'field_significance', 'funding_priority_alignment', 'societal_significance'},
    'complementarity articulation': {'expertise_complementarity', 'method_complementarity', 'resource_complementarity'},
    'coordination & decision practices': {'calls_for_decision', 'checks_consensus', 'invites_contribution', 'named_next_step', 'proposes_next_step', 'proposes_process', 'records_or_documents'},
    'epistemic bridging': {'connects_methods', 'reframes_cross_disciplinarily', 'translates_terminology'},
    'evaluation practices': {'compares_options', 'critiques_or_challenges', 'defends', 'devil_advocate', 'evaluates', 'extends_existing_idea', 'raises_concern', 'resolves_contradiction', 'setback_response_accepts_builds', 'setback_response_defends', 'setback_response_explores', 'setback_response_redirects', 'supports_or_validates'},
    'future-oriented language': {'named_next_step', 'specific_future_plan', 'vague_future_reference'},
    'idea management': {'attributes_to_other', 'claims_own_idea', 'combines_ideas', 'extends_existing_idea', 'proposes_new_idea', 'redirects_idea', 'returns_to_earlier_idea', 'synthesizes_contributions'},
    'idea novelty signal': {'novelty_recognized_other', 'novelty_recognized_self'},
    'idea ownership & attribution': {'attributes_to_other', 'challenges_attribution', 'claims_own_idea'},
    'information seeking': {'asks_clarifying_question', 'asks_factual_question', 'asks_for_elaboration', 'asks_for_opinion', 'asks_rhetorical_question', 'invites_contribution'},
    'integration practices': {'connects_methods', 'extends_existing_idea', 'frames_shared_solution', 'identifies_common_ground', 'resolves_contradiction', 'synthesizes_contributions'},
    'knowledge sharing': {'shares_data_or_findings', 'shares_domain_knowledge', 'shares_factual_knowledge', 'shares_method_or_approach', 'shares_personal_experience'},
    'participation dynamics': {'encourages_participation', 'gatekeeps', 'invites_contribution', 'redirects_speaker', 'summarizes_for_group', 'yields_floor'},
    'pronoun framing': {'ambiguous', 'individual_framing', 'joint_framing'},
    'relational climate': {'expresses_appreciation', 'expresses_enthusiasm', 'manages_tension'},
    'role anticipation': {'explicit_role_assignment', 'implicit_role_suggestion'},
    'setback response': {'setback_response_accepts_builds', 'setback_response_defends', 'setback_response_explores', 'setback_response_redirects'},
}

cn_alias = {
    'broader significance': 'broader scientific significance',
    'coordination and decision practices': 'coordination & decision practices',
    'idea ownership and attribution': 'idea ownership & attribution',
}
sc_alias = {'none_unclassified': 'none'}

files = 0
rows = 0
code_counts = Counter()
sub_counts = Counter()
pair_counts = Counter()
unexpected_code_name = Counter()

for f in root.rglob('*chunk*.json'):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    uas = d.get('utterance_annotations')
    if not isinstance(uas, list):
        continue
    files += 1
    for ua in uas:
        for c in ua.get('codes', []):
            cn = c.get('code_name')
            sc = c.get('subcode')
            if not isinstance(cn, str) and not isinstance(sc, str):
                continue
            rows += 1
            cn_n = (cn or '').strip().lower()
            sc_n = (sc or '').strip().lower()
            cn_n = cn_alias.get(cn_n, cn_n)
            sc_n = sc_alias.get(sc_n, sc_n)
            if cn_n and cn_n != 'none':
                code_counts[cn_n] += 1
            if sc_n and sc_n != 'none':
                sub_counts[sc_n] += 1
            if cn_n and sc_n:
                pair_counts[(cn_n, sc_n)] += 1
            if cn_n and cn_n != 'none' and cn_n not in exp:
                unexpected_code_name[cn_n] += 1

invalid_pairs = Counter()
for (cn, sc), n in pair_counts.items():
    if cn in exp and sc and sc != 'none' and sc not in exp[cn]:
        invalid_pairs[(cn, sc)] = n

missing_code_names = [k for k in exp if code_counts[k] == 0]

print('usable_chunk_files', files)
print('total_code_rows', rows)
print('distinct_code_names_seen', len(code_counts))
print('distinct_subcodes_seen', len(sub_counts))
print('unexpected_code_name_count', sum(unexpected_code_name.values()))
print('distinct_unexpected_code_names', len(unexpected_code_name))
print('invalid_pair_count', sum(invalid_pairs.values()))
print('distinct_invalid_pairs', len(invalid_pairs))
print('missing_expected_code_names', missing_code_names)

print('\nTop expected code_names:')
for cn, n in code_counts.most_common(20):
    print(cn, n)

print('\nTop unexpected code_names:')
for cn, n in unexpected_code_name.most_common(20):
    print(cn, n)

print('\nTop invalid pairs:')
for (cn, sc), n in invalid_pairs.most_common(30):
    print(f'{cn} -> {sc}: {n}')

out_dir = Path('/Users/maxchalekson/Downloads')
with (out_dir / 'audit_unexpected_code_names.csv').open('w') as fh:
    fh.write('code_name,count\n')
    for cn, n in unexpected_code_name.most_common():
        fh.write(f'"{cn}",{n}\n')

with (out_dir / 'audit_invalid_code_subcode_pairs.csv').open('w') as fh:
    fh.write('code_name,subcode,count\n')
    for (cn, sc), n in invalid_pairs.most_common():
        fh.write(f'"{cn}","{sc}",{n}\n')

print('\nwritten', out_dir / 'audit_unexpected_code_names.csv')
print('written', out_dir / 'audit_invalid_code_subcode_pairs.csv')
