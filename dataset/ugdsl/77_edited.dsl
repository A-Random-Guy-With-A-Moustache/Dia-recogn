N A Start 400 150 E W
N B Patient_arrives 400 250 B Y
E A B -
N C Registered_patient 400 350 D Y
E B C -
E C B -
N D Register_patient 380 450 B Y
E C D -
N E Available_nurse 500 350 D Y
E C E -
E D E -
N F Wait_for_available_nurse 500 450 B Y
E E F -
N G Record_health_condition 650 350 B Y
E E G -
N H Available_doctor 650 450 D Y
E G H -
N I Assign_patient_to_doctor 750 450 B Y
E H I -
N J Need_follow_up 750 550 D Y
E I J -
N K Arrange_appointment 850 550 B Y
E J K -
N L Give_patient_prescription 850 650 B Y
E J L -
N M Need_Medication 750 650 D Y
E L M -
E K M -
N N Patient_Leaves 750 750 B Y
E M N -
E L N -
E K N -
N O Stop 750 850 E W
E N O -
E M O -
E L O -
E K O -
N P Log_Successful_Care 900 750 B Y
E N P -