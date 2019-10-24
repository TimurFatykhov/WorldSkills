
# World Skills: Машинное Обучение
---
**Windows:**
<br>
Поиск -> Anaconda Prompt -> выполняем команды:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
python -m pip install --upgrade pip wheel setuptools
python -m pip install docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew
python -m pip install kivy.deps.gstreamer
python -m pip install kivy.deps.angle
python -m pip install pygame
python -m pip install kivy
pip install albumentations
```

Чтобы запустить графическое приложение для распознавания цифр:

- скачиваем репозиторий в папку Downloads
- разархивируем сжатый файл в Downloads
- выполняем в Anaconda Prompt:

```
cd "Downloads\WorldSkills-master\mnist recognition"
python app.py
```
