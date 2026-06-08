@echo off
cd /d "C:\Users\GramSC\Documents\weather"

:: Stage all changes
git add .

:: Commit with a timestamp
git commit -m "Auto-update: %DATE% %TIME%"

:: Push to GitHub
git push origin main

echo Done!