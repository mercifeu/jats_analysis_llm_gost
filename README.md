# jats_analysis_llm_gost

1. Склонировать репозиторий.
  ```git clone https://github.com/mercifeu/jats_analysis_llm_gost.git```
3. Создать виртаульное окружение:
  ```python -m venv venv```
4. Установить библиотеки:
  ```pip install -r requirements.txt```
5. Создать файл .env и вставить API:
  ```DEEPSEEK_API_KEY=sk-any```

Или для удобства создать run.bat:

```bat
  @echo off
  cd /d "%~dp0"
  
  REM Проверка, существует ли уже виртуальное окружение
  if not exist "venv\Scripts\activate.bat" (
      echo.
      echo Виртуальное окружение не найдено. Создаю...
      python -m venv venv
      
      REM Опционально: если есть файл requirements.txt, скрипт сам установит библиотеки
      if exist "requirements.txt" (
          echo Установка зависимостей из requirements.txt...
          call venv\Scripts\pip.exe install -r requirements.txt
      )
  )
  
  REM Активирует виртуальное окружение
  echo Активация venv...
  call venv\Scripts\activate.bat
  
  REM Запускает программу
  echo.
  echo Запуск main.py...
  python main.py
  
  REM Пауза, чтобы окно консоли не закрылось мгновенно после завершения работы
  echo.
  echo Программа завершена.
  pause
```

В качестве proxy api рекомендуется:

https://github.com/ForgetMeAI/FreeDeepseekAPI

#989I34FI9Kl;ff4j908l;k-904f
