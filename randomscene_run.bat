@echo off
@REM run random scenejson

@REM @REM prm
@REM set name=lowfluid-S-r
@REM set name=cconv-scene-r
@REM set name=cconv-sceneless-r



@REM set start=10
@REM set end=100

@REM for /l %%i in (1,1,5) do (echo %%i) 
@REM echo  %name%FFF
@REM pause

@REM for /l %%i in (%start%,1,%end%) do (C:/Users/123/.conda/envs/ev1/python.exe .\run_simulation.py --scene_file .\randomScene\%name%%%i.json) 


@REM -----------------cconv scene----------------------

set name=scene
set start=41
set end=100
for /l %%i in (%start%,1,%end%) do (C:/Users/123/.conda/envs/ev1/python.exe .\run_simulation.py --scene_file .\data\scenes\cconvobjnpy.json --cconvsceneidx %%i) 
