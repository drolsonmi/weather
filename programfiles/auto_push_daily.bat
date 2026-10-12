@echo off
cd /d "C:\Users\GramSC\Documents\weather"

:: Update local repo
git pull

:: Clear garbage
git gc --force

:: Copy data files
xcopy /I /Y "C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat" "C:\Users\GramSC\Documents\weather\data"
xcopy /I /Y "C:\Campbellsci\LoggerNet\Snow Weather_FifteenMin.dat" "C:\Users\GramSC\Documents\weather\data"
xcopy /I /Y "C:\Campbellsci\LoggerNet\Snow Weather_Daily.dat" "C:\Users\GramSC\Documents\weather\data"

:: Stage all changes
git add "C:\Users\GramSC\Documents\weather\data\Snow Weather_FiveMin.dat"
git add "C:\Users\GramSC\Documents\weather\data\Snow Weather_FifteenMin.dat"
git add "C:\Users\GramSC\Documents\weather\data\Snow Weather_Daily.dat"

:: Commit with a timestamp
git commit -m "Auto-update: %DATE% %TIME%"

:: Push to GitHub
git push origin main

echo Done!