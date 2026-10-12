"""The README's metrics block is generated, never hand-edited."""
import json

import pytest

from modules import update_readme as U


def _metrics():
    return json.load(open(U.METRICS_PATH, encoding='utf-8'))


def test_committed_readme_matches_committed_metrics():
    """Fails if someone edits the generated block by hand or forgets to regenerate it."""
    text = open(U.README_PATH, encoding='utf-8').read()
    block = text.split(U.START)[1].split(U.END)[0].strip()
    assert block == U.render(_metrics()).strip()


def test_update_only_touches_marked_block(tmp_path):
    readme = tmp_path / 'README.md'
    readme.write_text(f'# Title\nhand-written intro\n{U.START}\nold numbers\n{U.END}\nhand-written outro\n', encoding='utf-8')
    assert U.update(str(readme), U.METRICS_PATH) is True
    out = readme.read_text(encoding='utf-8')
    assert out.startswith('# Title\nhand-written intro\n') and out.endswith('hand-written outro\n')
    assert 'old numbers' not in out and f"{_metrics()['matches']:,}" in out
    assert U.update(str(readme), U.METRICS_PATH) is False  # idempotent


def test_missing_markers_fail_loudly(tmp_path):
    readme = tmp_path / 'README.md'
    readme.write_text('# No markers here\n', encoding='utf-8')
    with pytest.raises(ValueError):
        U.update(str(readme), U.METRICS_PATH)


def test_render_handles_metrics_without_optional_sections():
    m = _metrics()
    for key in ('decision_quality', 'calibration'):
        m.pop(key, None)
    out = U.render(m)
    assert 'Log loss' in out and 'winning picks' not in out
