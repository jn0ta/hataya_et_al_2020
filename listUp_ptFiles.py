import os, pathlib

def listUp_ptFiles(_todays_date, _polDictFilename):
    # directory preparation
    _theEnvVar = os.environ['LISTUP_PTFILES'] # defined by shell script exeMain
    _path = pathlib.Path('../../polDicts/'+str(_todays_date.date()))
    if not _path.exists(): _path.mkdir(parents=True)
    
    # file preparation
    _filename = "polDictsList-"+_theEnvVar+".txt"
    _path = pathlib.Path('../../polDicts/'+str(_todays_date.date())+"/"+_filename)
    if not _path.exists(): _path.touch()

    with open(_path, 'a', encoding="utf-8") as _f:
        _f.write(_polDictFilename+",")
