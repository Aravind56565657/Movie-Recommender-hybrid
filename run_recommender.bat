@echo off
REM Run the Movie Recommender System

echo Select mode to run:
echo 1. Data Preprocessing (data)
echo 2. Collaborative Filtering (als)
echo 3. Demographic Filtering (demographic)
echo 4. Content-Based Filtering (content)
echo 5. Full Pipeline (full)
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto data
if "%choice%"=="2" goto als
if "%choice%"=="3" goto demographic
if "%choice%"=="4" goto content
if "%choice%"=="5" goto full
goto end

:data
echo Running Data Preprocessing...
python main.py --mode data
goto end

:als
echo Running Collaborative Filtering (ALS)...
python main.py --mode als
goto end

:demographic
echo Running Demographic Filtering...
python main.py --mode demographic
goto end

:content
echo Running Content-Based Filtering...
python main.py --mode content
goto end

:full
echo Running Full Pipeline...
python main.py --mode full
goto end

:end
pause
