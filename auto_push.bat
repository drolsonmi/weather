@echo off
cd /d "C:\Users\GramSC\Documents\weather"

:: Copy data files
xcopy /E /I /Y "C:\Campbellsci\LoggerNet\Snow Weather_Daily.dat" "C:\Users\GramSC\Documents\weather\data"
xcopy /E /I /Y "C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat" "C:\Users\GramSC\Documents\weather\data"
xcopy /E /I /Y "C:\Campbellsci\LoggerNet\Snow Weather_FifteenMin.dat" "C:\Users\GramSC\Documents\weather\data"

:: Stage all changes
git add .

:: Commit with a timestamp
git commit -m "Auto-update: %DATE% %TIME%"

:: Push to GitHub
git push origin main

echo Done!