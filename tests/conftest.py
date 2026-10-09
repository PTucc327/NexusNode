import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Never hit Riot's CDN from tests; use the bundled Data Dragon snapshot.
os.environ['NEXUSNODE_OFFLINE'] = '1'


@pytest.fixture(scope='session', autouse=True)
def project_root():
    """Run every test from the project root, where the data paths resolve."""
    previous = os.getcwd()
    os.chdir(ROOT)
    yield ROOT
    os.chdir(previous)


@pytest.fixture(scope='session')
def engine(project_root):
    from modules.engine import DraftingEngine
    return DraftingEngine()


ALLIES = {'TOP': 'Gnar', 'JUNGLE': 'XinZhao', 'MIDDLE': 'Ahri', 'SUPPORT': 'Rakan'}
ENEMIES = {'TOP': 'Jayce', 'JUNGLE': 'Nidalee', 'MIDDLE': 'Syndra', 'BOTTOM': 'Caitlyn', 'SUPPORT': 'Lux'}
