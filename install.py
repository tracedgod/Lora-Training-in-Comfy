import subprocess
import os.path

class Install:
    sd_scripts_path = os.path.join(os.path.dirname(__file__), "sd-scripts")

    @staticmethod
    def check_install():
        if not os.path.exists(Install.sd_scripts_path):
            Install.install()

    @staticmethod
    def install():
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=os.path.dirname(__file__),
            shell=True
        )

    @staticmethod
    def clone():
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--branch", "sd3",
                "https://github.com/kohya-ss/sd-scripts", "sd-scripts"
            ],
            cwd=os.path.dirname(__file__)
        )