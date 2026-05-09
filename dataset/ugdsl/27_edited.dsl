N A Start 500 50 E P
N B Patient_arrives 500 150 B Y
E A B -
N C Registered_patient 500 250 D O
E B C -
E C B -
N D Register_patient 500 350 B Y
E C D -
N E Available_nurse 800 250 D O
E C E -
E D E -
N F Wait_for_available_nurse 800 350 B Y
E E F -
N G Record_health_condition 800 450 B Y
E E G -
N H Available_doctor 800 550 D O
E G H -
N I Assign_patient_to_doctor 800 650 B Y
E H I -
N J Need_follow_up 800 750 D O
E I J -
N K Arrange_appointment 800 850 B Y
E J K -
N L Need_Medication 800 950 D O
E K L -
N M Give_patient_prescription 800 1050 B Y
E L M -
N N Patient_Leaves 800 1150 B Y
E M N -
E L N -
N O Stop 800 1250 E P
E N O -
N Q Log_Successful_Care 800 1150 D O
E N Q -