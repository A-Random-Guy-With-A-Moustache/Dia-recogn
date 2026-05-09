N A Start 580 150 E G
N B Declare_variables_a_b_and_c 580 250 B B
N C Assign_a_b_and_c 580 350 B B
N D Is_a>b? 580 450 D B
N E Print_b 300 600 I B
N F Print_c 580 600 I B
N G Print_a 800 600 I B
N H Stop 580 700 B R

E A B -
E B C -
E C D -
E D E False
E D F True
E D G True
E E H -
E F H -
E G H -