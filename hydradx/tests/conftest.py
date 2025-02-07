from hypothesis import settings
settings.register_profile("ci", deadline=None, print_blob=True)
settings.load_profile("ci")
