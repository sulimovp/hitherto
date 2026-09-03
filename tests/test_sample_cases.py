from casefile.eval.sample_cases import load_sample_cases


def test_sample_cases_yaml_loads():
    cases = load_sample_cases()
    assert len(cases) >= 6
    assert "pytorch-masked" in cases
    assert cases["numpy-ma"].path == "numpy/ma"
    assert cases["pytorch-masked"].blurb


def test_all_cases_have_required_ui_fields():
    for case in load_sample_cases().values():
        assert case.label
        assert case.question
        assert case.repo
        preset = case.preset_dict()
        assert preset["blurb"] == case.blurb
