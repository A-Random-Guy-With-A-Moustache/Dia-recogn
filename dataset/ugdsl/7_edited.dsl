N A Start 500 50 G
N B Declare_variables_a_b_and_c 500 150 B B
N C Assign_a_b_and_c 500 250 I B
N D Is_a>b? 500 350 D Y
N E Is_c>b? 300 450 D Y
N F Print_b 300 550 I B
N G Is_a>c? 800 450 D Y
N H Print_c 500 550 I B
N I Print_a 800 550 I B
N J Stop 500 650 C R

E A B -
E B C -
E C D -
E D E False
E D G True
E E F False
E G H False
E G I True
E F J -
E H J -
E I J -