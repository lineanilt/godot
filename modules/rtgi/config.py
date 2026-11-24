def can_build(env, platform):
    return platform in ["x11", "windows"]


def configure(env):
    pass


def get_doc_classes():
    return [
        "RTGI",
    ]


def get_doc_path():
    return "doc_classes"
