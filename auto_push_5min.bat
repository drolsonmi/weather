@echo off
cd /d "C:\Users\GramSC\Documents\weather"

:: Copy data files
xcopy /I /Y "C:\Campbellsci\LoggerNet\Snow Weather_FiveMin.dat" "C:\Users\GramSC\Documents\weather\data"
xcopy /I /Y "C:\Campbellsci\LoggerNet\Snow Weather_FifteenMin.dat" "C:\Users\GramSC\Documents\weather\data"

:: New Image
C:\Users\GramSC\.virtualenvs\Wx\Scripts\python.exe C:\Users\GramSC\Documents\weather\images\WxImage.py

:: Stage all changes
git add "C:\Users\GramSC\Documents\weather\data\Snow Weather_FiveMin.dat"
git add "C:\Users\GramSC\Documents\weather\data\Snow Weather_FifteenMin.dat"
git add "C:\Users\GramSC\Documents\weather\images\weather_image.png"

:: Commit with a timestamp
git commit -m "Auto-update: %DATE% %TIME%"

:: Push to GitHub
git push origin main

echo Done!