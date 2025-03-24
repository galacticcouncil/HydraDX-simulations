def test_hsm():
    from hydradx.apps.stability_module.hsm import hollar_burned
    assert hollar_burned > 400000