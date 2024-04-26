@echo off
color a

echo Installing modules...

pip install numpy > NUL

pip install opencv-python > NUL

echo Modules installed.

echo starting Face rec.py
timeout /t 5 /nobreak > NUL
echo Program started successfully

python lol.py > NUL

echo Program started successfully