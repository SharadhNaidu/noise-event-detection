import pytest

# Test files that need downloaded assets (BEATs / ATST-F weights in ext/PretrainedSED/resources, member checkpoints in
# $INDOML_ROOT/ckpt) or built audio / feature stores in $INDOML_ROOT. They are deselected by default (pytest.ini);
# run them with `pytest -m data` after scripts/download_assets.py and the store builds in the README.
DATA_FILES = {
    "test_ast_wiring.py", "test_aux_generic.py", "test_mt_aux.py", "test_submit_ast.py",
    "test_all_channels.py", "test_ast_warmstart.py", "test_atst_ft.py", "test_beats_ft.py",
    "test_conformer_head.py", "test_dual_aux.py", "test_submit_ens.py",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.name in DATA_FILES:
            item.add_marker(pytest.mark.data)
