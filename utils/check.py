import platform
import subprocess
from pathlib import Path
from utils.utils import colorstr


def check_online():
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessible
        return True
    except OSError:
        return False


def check_git_status():
    # Recommend 'git pull' if code is out of date
    print(colorstr('github: '), end='')
    try:
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        assert check_online(), 'skipping check (offline)'

        cmd = 'git fetch && git config --get remote.origin.url'
        url = subprocess.check_output(cmd, shell=True).decode().strip().rstrip('.git')  # github repo url
        branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()  # checked out
        n = int(subprocess.check_output(f'git rev-list {branch}..origin/master --count', shell=True))  # commits behind
        if n > 0:
            s = f"⚠️ WARNING: code is out of date by {n} commit{'s' * (n > 1)}. " \
                f"Use 'git pull' to update or 'git clone {url}' to download latest."
        else:
            s = f'up to date with {url} ✅'
        print(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    except Exception as e:
        print(e)


def check_requirements(file='requirements.txt', exclude=()):
    import pkg_resources
    requirements = [f'{x.name}{x.specifier}' for x in pkg_resources.parse_requirements(Path(file).open())
                    if x.name not in exclude]
    pkg_resources.require(requirements)
