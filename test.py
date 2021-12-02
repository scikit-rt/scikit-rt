from skrt import Patient
p = Patient("~/Work/HeadAndNeck/dicom/VT1_H_03F693K1")
print(p.studies[0].ct_structure_sets[-1])
