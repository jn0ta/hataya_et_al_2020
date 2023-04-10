import os, pathlib, torch

def savePolicy(_policy, _policySaveFilename, _locationOfTheDir):
    # directory preparation
    _theEnvVar = os.environ['DIRECTORY_RESULTFILES'] # defined by shell script exeMain
    #_path = pathlib.Path(_locationOfTheDir+str(_todays_date.date()))
    _path = pathlib.Path(_locationOfTheDir+_theEnvVar)
    if not _path.exists(): _path.mkdir(parents=True)
    
    _path = pathlib.Path(_locationOfTheDir+_theEnvVar+"/"+_policySaveFilename)
    torch.save(_policy.state_dict(), _path) # output the .pt file outside of the directory "output"