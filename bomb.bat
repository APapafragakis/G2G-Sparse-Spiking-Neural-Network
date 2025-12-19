@echo off
setlocal enabledelayedexpansion

REM =========================
REM GLOBAL DEFAULTS
REM =========================
set EPOCHS=20
set T_MAIN=50
set BS=256
set ENC_SCALE=1.0
set ENC_BIAS=0.0
set SPARSITY=static

REM p' list for main sweep
set P_LIST_MAIN=0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50

REM =========================
REM ROOT RESULTS FOLDER
REM =========================
set ROOT=results
if not exist %ROOT% mkdir %ROOT%

REM ============================================================
REM PHASE 1 — MAIN EXPERIMENT
REM dataset × model × p' (current encoding)
REM ============================================================
echo.
echo ===== PHASE 1: MAIN EXPERIMENT =====

set PHASE1=%ROOT%\phase1_main
if not exist %PHASE1% mkdir %PHASE1%

for %%D in (fashionmnist cifar10 cifar100) do (
  mkdir %PHASE1%\%%D
  for %%M in (dense index random mixer) do (
    for %%P in (%P_LIST_MAIN%) do (
      echo [PHASE 1] %%D | %%M | p'=%%P

      python train.py ^
        --dataset %%D ^
        --model %%M ^
        --p_inter %%P ^
        --epochs %EPOCHS% ^
        --T %T_MAIN% ^
        --batch_size %BS% ^
        --enc current ^
        --enc_scale %ENC_SCALE% ^
        --enc_bias %ENC_BIAS% ^
        --sparsity_mode %SPARSITY% ^
        > %PHASE1%\%%D\%%M_pinter_%%P.txt 2>&1
    )
  )
)

REM ============================================================
REM PHASE 2 — ENCODING ROBUSTNESS
REM rate vs current
REM ============================================================
echo.
echo ===== PHASE 2: ENCODING ROBUSTNESS =====

set PHASE2=%ROOT%\phase2_encoding
if not exist %PHASE2% mkdir %PHASE2%

for %%D in (fashionmnist cifar10) do (
  mkdir %PHASE2%\%%D
  for %%M in (index mixer) do (
    for %%P in (0.15 0.25) do (
      for %%E in (current rate) do (
        echo [PHASE 2] %%D | %%M | p'=%%P | enc=%%E

        python train.py ^
          --dataset %%D ^
          --model %%M ^
          --p_inter %%P ^
          --epochs %EPOCHS% ^
          --T %T_MAIN% ^
          --batch_size %BS% ^
          --enc %%E ^
          --enc_scale %ENC_SCALE% ^
          --enc_bias %ENC_BIAS% ^
          --sparsity_mode %SPARSITY% ^
          > %PHASE2%\%%D\%%M_pinter_%%P_enc_%%E.txt 2>&1
      )
    )
  )
)

REM ============================================================
REM PHASE 3 — TIME WINDOW SENSITIVITY
REM T sweep (current injection)
REM ============================================================
echo.
echo ===== PHASE 3: TIME WINDOW SENSITIVITY =====

set PHASE3=%ROOT%\phase3_time
if not exist %PHASE3% mkdir %PHASE3%

mkdir %PHASE3%\fashionmnist

for %%T in (10 20 50) do (
  echo [PHASE 3] FashionMNIST | Index | T=%%T

  python train.py ^
    --dataset fashionmnist ^
    --model index ^
    --p_inter 0.25 ^
    --epochs %EPOCHS% ^
    --T %%T ^
    --batch_size %BS% ^
    --enc current ^
    --enc_scale %ENC_SCALE% ^
    --enc_bias %ENC_BIAS% ^
    --sparsity_mode %SPARSITY% ^
    > %PHASE3%\fashionmnist\index_T_%%T.txt 2>&1
)

echo.
echo ===== ALL EXPERIMENTS FINISHED =====
