N A НАЧАЛО 500 100 C W
N B ВВОД_A_B_C 500 200 D W
N C S 800 200 D W
N D X 900 200 D W
N E X1_X2 900 300 D W
N F ВЫВОД_X 900 400 D W
N G ВЫВОД_X1_X2 900 500 D W
N H КОНЦ 900 600 C W

E A B +
E B C =
E C D +
E D E +
E E F +
E F G +
E G H +

U B C -
U C D -
U D E -

U E F -
U F G -
U G H -