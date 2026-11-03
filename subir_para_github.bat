@echo off
echo ========================================================
echo Sincronizacao do Repositorio do Projeto 2 com o GitHub
echo ========================================================
echo.
echo 1. Autenticando com o GitHub via navegador...
gh auth login -w -p https
echo.
echo 2. Criando o repositorio no GitHub (caso ainda nao exista)...
gh repo create lc-sc-reasoning-overthinking --public --source=. --remote=origin --description "Dynamic Overthinking Mitigation in Small Reasoning LLMs via Shannon Entropy Early Stopping (UnB PPGI0034 - ICLR 2026)"
echo.
echo 3. Enviando branches e tags para o GitHub...
git push -u origin main
echo.
echo ========================================================
echo Repositorio enviado com sucesso para o GitHub!
echo ========================================================
pause
