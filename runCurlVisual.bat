@REM 一次只处理一帧，反复调用curvisual
@echo off


@REM prm
set emitstart=1
set emitend=50


FOR /L %%i IN (%emitstart%,1,%emitend%) DO (  
        C:/Users/123/.conda/envs/ev1/python.exe d:/CODE/MCVSPH-FORK/curlvisual.py --lv=%%i --rv=%%i
        echo -----------%%i------------
)  

@REM C:/Users/123/.conda/envs/ev1/python.exe d:/CODE/MCVSPH-FORK/curlvisual.py --lv=%emitend% --rv=1000
