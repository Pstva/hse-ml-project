
## Используемые инструменты

+ линтер pycodestyle
+ линтер PyFlakes
+ форматтер Black


### pycodestyle (2.6.0)


Этот инструмент был выбран для того, чтобы проверить, насколько мой код по стилю соответсвует стандарту PEP. 

Запускался так:

```sh
pycodestyle --statistics -qq src
```

#### Вывод:

```sh
1       E115 expected an indented block (comment)
1       E128 continuation line under-indented for visual indent
22      E251 unexpected spaces around keyword / parameter equals
8       E261 at least two spaces before inline comment
1       E265 block comment should start with '# '
1       E301 expected 1 blank line, found 0
6       E303 too many blank lines (2)
1       E306 expected 1 blank line before a nested definition, found 0
23      E501 line too long (104 > 79 characters)
19      W291 trailing whitespace
58      W293 blank line contains whitespace
4       W391 blank line at end of file

```

Выведено довольно много нарушений PEP.

### pyflakes (2.4.0)

Инструмент выбран, чтобы посмотреть на возможные ошибки в скрипте

```sh
pyflakes src/
```
#### Вывод:

```sh
src/metrics.py:3:1 'matplotlib' imported but unused
src/metrics.py:4:1 'typing.NoReturn' imported but unused
src/metrics.py:4:1 'typing.List' imported but unused
src/main.py:2:1 redefinition of unused 'plot_precision_recall' from line 2
src/main.py:2:1 'src.metrics.get_precision_recall_accuracy' imported but unused
src/main.py:3:1 'src.kdtree.Node' imported but unused
src/main.py:3:1 'src.kdtree.KDTree' imported but unused
src/main.py:4:1 'src.knn.KNearest' imported but unused
src/kdtree.py:2:1 'typing.NoReturn' imported but unuse
```

Как можно заметить, у меня было импортировано несколько функций, которые я не использую в дальнейшем. 
Я убрала их импорт из скриптов, после чего инструмент больше не выдавал мне ошибок.

### Black (19.10b0)


Так как у меня было довольно много стилистических нарушений в коде, я решила использовать форматтер Black
с дефолтными настройками для того, чтобы это исправить.

```sh
black src/
```

#### Вывод:

```sh
reformatted src/main.py
reformatted src/knn.py
reformatted src/read_prepare_data.py
reformatted src/metrics.py
reformatted src/kdtree.py
All done! ✨ 🍰 ✨
5 files reformatted.
```

