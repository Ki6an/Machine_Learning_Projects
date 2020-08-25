from music21 import environment

us = environment.UserSettings()

for key in sorted(us.keys()):
    print(key)

us['musicxmlPath'] = "/Program Files/MuseScore 3/bin/MuseScore3.exe"
print(us['musicxmlPath'])