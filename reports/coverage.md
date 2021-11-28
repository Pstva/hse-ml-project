## Репорт о покрытии скрипта тестами

Команда:

```sh
coverage run -m pytest && coverage report -m
```

Вывод: 

```
============================= test session starts ==============================
platform linux -- Python 3.8.8, pytest-6.2.3, py-1.10.0, pluggy-0.13.1
rootdir: /home/alena/gits/ml-project (копия)
plugins: anyio-2.2.0
collected 13 items                                                             

test_scripts.py .............                                            [100%]

============================== 13 passed in 9.72s ==============================
Name                       Stmts   Miss  Cover   Missing
--------------------------------------------------------
src/kdtree.py                 80      0   100%
src/knn.py                    26      0   100%
src/metrics.py                77     57    26%   47-77, 81-114
src/read_prepare_data.py      23      0   100%
test_scripts.py              163      0   100%
--------------------------------------------------------
TOTAL                        369     57    85%
```

В скрипте *metrics.py* есть несколько функций с рисованием графиков, которые не тестировались, из-за такой маленький процент покрытия тестами для этого файла.

