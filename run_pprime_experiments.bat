@echo off
echo =========================================
echo   Running IndexSNN Experiments (p')
echo   epochs=10 | batch_size=128 | T=25
echo =========================================

REM ---- Run 1: p_inter = 0.00 ----
echo.
echo ----- Running p_inter = 0.00 -----
python train.py --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.00
echo ------------------------------------

REM ---- Run 2: p_inter = 0.10 ----
echo.
echo ----- Running p_inter = 0.10 -----
python train.py --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.10
echo ------------------------------------

REM ---- Run 3: p_inter = 0.25 ----
echo.
echo ----- Running p_inter = 0.25 -----
python train.py --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.25
echo ------------------------------------

echo.
echo All experiments completed!
pause
