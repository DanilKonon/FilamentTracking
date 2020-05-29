# FilamentTracking

Программа для трекинга актиновых филаментов. 

# Installation

1. Скопируйте репозиторий к себе на компьютер 
2. С помощью pip установите все зависимости (лучше в новой среде с python3.7): 

```
pip install -r requirements.txt
```
Также можно с помощью conda сразу создать и среду, и установить в неё все необходимые модули. 
```
conda create --name <env> --python=3.7 --file requirements_conda.txt
```

# Usage 

1. Сначала необходимо запустить main.py, указав путь к последовательности филаментов в tiff формате
2. После этого нужно запустить fil_processing.py, снова указав путь к последовательности

Пример. Сначала используем main.py.

```
python main.py /Users/danilkononykhin/PycharmProjects/Filaments/Actin_filaments/Motility_Mar.19__tiff_mdf/25031911.tif   
```
Теперь используем fil_processing.py.
```
python fil_processing.py /Users/danilkononykhin/PycharmProjects/Filaments/Actin_filaments/Motility_Mar.19__tiff_mdf/25031911.tif   
```