

@REM -----------------↓cconv Specific scene↓----------------------
@REM 泛化场景
set name=scene
set start=1
set end=1
for /l %%i in (%start%,1,%end%) do (C:/Users/123/.conda/envs/ev1/python.exe .\run_simulation.py --scene_file .\data\scenes\cconvsp.json --cconvsceneidx %%i) 
