"""Guards that must never regress before/after going public."""
import glob
import json
import re
import subprocess

import pytest

RIOT_KEY_PATTERN = re.compile(r'RGAPI-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-')


def tracked_files():
    out = subprocess.run(['git', 'ls-files'], capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f]


def test_no_riot_api_key_in_tracked_files():
    leaks = []
    for path in tracked_files():
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if RIOT_KEY_PATTERN.search(f.read()):
                    leaks.append(path)
        except (IsADirectoryError, FileNotFoundError):
            continue
    assert leaks == [], f'Riot API key found in tracked files: {leaks}'


def test_env_file_is_ignored():
    result = subprocess.run(['git', 'check-ignore', '-q', '.env'])
    assert result.returncode == 0, '.env must be git-ignored'


def test_dependencies_are_pinned():
    lines = [ln.strip() for ln in open('requirements.txt') if ln.strip() and not ln.startswith('#')]
    unpinned = [ln for ln in lines if '==' not in ln]
    assert unpinned == [], f'unpinned dependencies: {unpinned}'


def test_github_actions_pinned_to_commit_sha():
    unpinned = []
    for wf in glob.glob('.github/workflows/*.yml'):
        for line in open(wf, encoding='utf-8'):
            m = re.search(r'uses:\s*([^\s#]+)', line)
            if m and not re.search(r'@[0-9a-f]{40}$', m.group(1)):
                unpinned.append(f'{wf}: {m.group(1)}')
    assert unpinned == [], f'actions not pinned to a commit SHA: {unpinned}'


def test_xsrf_protection_not_disabled():
    for path in ['.devcontainer/devcontainer.json', *glob.glob('.streamlit/*.toml')]:
        try:
            text = open(path, encoding='utf-8').read()
        except FileNotFoundError:
            continue
        assert 'enableXsrfProtection false' not in text and 'enableXsrfProtection = false' not in text, path


def test_model_loads_without_arbitrary_code():
    """weights_only=True: a tampered model file can't execute code on load."""
    import torch
    art = torch.load('data/processed/nexus_model.pt', weights_only=True)
    assert {'champions', 'embeddings', 'role_strength'} <= set(art)


def test_training_data_has_no_player_identifiers():
    import pandas as pd
    for path in ['data/raw/league_match_data.csv', 'data/processed/cleaned_league_match_data.csv']:
        cols = {c.lower() for c in pd.read_csv(path, nrows=0).columns}
        assert not cols & {'puuid', 'summoner_name', 'summonerid', 'riot_id', 'game_name', 'tag_line'}, path


@pytest.mark.parametrize('field', ['matches', 'validation', 'patches'])
def test_published_metrics_present(field):
    assert field in json.load(open('data/processed/model_metrics.json'))
