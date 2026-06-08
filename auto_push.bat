@echo off
cd /d "C:\Users\GramSC\Documents\weather"

:: Copy data files
cp "C:\Campbellsci\LoggerNet\Snow Weather_Daily.dat" .\data
cp "C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat" .\data
cp "C:\Campbellsci\LoggerNet\Snow Weather_FifteenMin.dat" .\data

:: Stage all changes
git add .

:: Commit with a timestamp
git commit -m "Auto-update: %DATE% %TIME%"

:: Push to GitHub
git push origin main

echo Done!