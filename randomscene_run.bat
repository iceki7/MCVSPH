@echo off
@REM run random scenejson

@REM prm
set name=lowfluid-S-r
set name=cconv-scene-r
set name=cconv-sceneless-r

set start=0
set end=100

@REM for /l %%i in (1,1,5) do (echo %%i) 
@REM echo  %name%FFF
@REM pause

for /l %%i in (%start%,1,%end%) do (C:/Users/123/.conda/envs/ev1/python.exe .\run_simulation.py --scene_file .\randomScene\%name%%%i.json) 
