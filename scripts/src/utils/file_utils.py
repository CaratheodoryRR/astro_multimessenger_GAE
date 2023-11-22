
from pathlib import Path

def not_empty_file(filename):
    return bool(Path(filename).stat().st_size)


# Checks if the directory exists. If it does not exist, we create it
def check_dir(dirName):
    dirName = Path(dirName)
    if not dirName.exists():
        dirName.mkdir(parents=True)
        print("New directory created: {}".format(dirName.resolve()))


# Similar to rm *.txt
def del_by_extension(parentDir, exts, recursive=False):
    
    pattern = '**/' if recursive else ''
    for fName in Path(parentDir).glob("{}*".format(pattern)):
        if fName.suffix in exts: 
            fName.unlink()
        