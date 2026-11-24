@echo off
setlocal
cd /d "%~dp0"
if "%PYTHON%"=="" set PYTHON=python
echo == residue self-check == & %PYTHON% -m checks.verify_residue
echo == structure self-check == & %PYTHON% -m checks.verify_structure
echo == run 1 == & %PYTHON% -m runs.run1_ablation
echo == run 2 == & %PYTHON% -m runs.run2_reconciliation
echo == run 3 == & %PYTHON% -m runs.run3_faithfulness
echo == tests == & %PYTHON% -m checks.run_tests
echo Done. See .\outputs\
endlocal
