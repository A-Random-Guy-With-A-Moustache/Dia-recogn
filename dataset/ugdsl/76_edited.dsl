N A Начало 580 220 E W
N B Ввести_массив 600 300 I W
E A B -
N C max_:=_массив(1) 600 390 B W
E B C -
N D i:=2,_k 600 480 D W
E C D -
N E max_<=_массив(i) 600 570 D W
E D E -
U E F - 
N G max_:=_массив(i) 600 660 B W
E F G -
E E G Да
E E G Нет
E G H -
N H Вывести_max 600 750 B W
E G H -
N I Конец 600 840 E W
E H I -