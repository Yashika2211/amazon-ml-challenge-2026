from metrics import f05, macro_f05


def test_readme_example():
    s = f05({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
    assert abs(s - 0.714) < 1e-3, s


def test_empty_rules():
    assert f05(set(), set()) == 1.0
    assert f05(set(), {"S2-1"}) == 0.0
    assert f05({"S2-1"}, set()) == 0.0
    assert f05({"S2-1"}, {"S2-2"}) == 0.0


def test_macro():
    t = {"a": {"x"}, "b": set()}
    p = {"a": {"x"}}
    assert macro_f05(t, p) == 1.0


if __name__ == "__main__":
    test_readme_example(); test_empty_rules(); test_macro()
    print("metric tests passed")
