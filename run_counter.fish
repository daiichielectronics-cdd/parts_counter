#!/usr/bin/env fish
echo "電子部品カウントシステムを起動中..."
cd (dirname (status --current-filename))
./venv/bin/python main.py 