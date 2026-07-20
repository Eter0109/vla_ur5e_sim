@echo off
start "P3 ablation watcher" /b powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\vla_ur5e_sim\scripts\watch_p3_ablation.ps1" 1>"D:\vla_ur5e_sim\outputs\p3_ablation_watcher.log" 2>"D:\vla_ur5e_sim\outputs\p3_ablation_watcher.error.log"
