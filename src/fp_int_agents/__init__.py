from fp_int_agents.app import FPIntAgentsApp, _check_project_root


def main() -> None:
    _check_project_root()
    FPIntAgentsApp().run()
