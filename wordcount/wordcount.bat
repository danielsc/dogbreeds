@echo off
rem redirect the output of wc.exe to output file
set in=%1
set out=%2
if "%out%"=="" set out=out_%in%
echo           Lines	Words	Characters	File > %out%
wc.exe -lwc %in% >>%out%