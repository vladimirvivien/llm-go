package workflow

type ExtractedRequest struct {
	RequestNumber string `json:"request_number" description:"PA / request number printed at the top of the form"`
	PatientName   string `json:"patient_name" description:"Patient's full name"`
	PatientDOB    string `json:"patient_dob" description:"Patient date of birth in YYYY-MM-DD"`
	MemberID      string `json:"member_id" description:"Member or patient ID"`
	Prescriber    string `json:"prescriber" description:"Prescriber name and credentials"`
	NPI           string `json:"npi" description:"Prescriber NPI number"`
	Drug          string `json:"drug" description:"Requested medication name"`
	Dosage        string `json:"dosage" description:"Requested dosage and frequency"`
	Diagnosis     string `json:"diagnosis" description:"Patient diagnosis"`
	ICD10         string `json:"icd10" description:"ICD-10 code"`
	Justification string `json:"justification" description:"Clinical justification narrative"`
}
