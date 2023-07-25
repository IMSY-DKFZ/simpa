import subprocess


# retrieves the latest git commit hash and saves it into a file
def build():
    hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    f = open("GIT_COMMIT_HASH", "w")
    f.write(hash)
    f.close()


if __name__ == '__main__':
    build()
