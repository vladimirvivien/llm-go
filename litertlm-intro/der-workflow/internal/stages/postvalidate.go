package stages

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/vladimirvivien/exception-workflow/internal/store"
	"github.com/vladimirvivien/exception-workflow/internal/workflow"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type PostValidateResult struct {
	Complete             bool     `json:"complete" description:"Are all required fields present and well-formed?"`
	MissingFields        []string `json:"missing_fields" description:"List of required fields that are empty or malformed"`
	NarrativeMatchesDrug bool     `json:"narrative_matches_drug" description:"Does the clinical justification reference or relate to the requested medication?"`
	DiagnosisPlausible   bool     `json:"diagnosis_plausible" description:"Does the diagnosis plausibly match the requested medication's typical use?"`
	Reason               string   `json:"reason" description:"One short sentence explaining the verdict"`
}

var (
	npiRegex   = regexp.MustCompile(`^\d{10}$`)
	icd10Regex = regexp.MustCompile(`^[A-Z]\d{2}(\.\d{1,4})?$`)
	dobRegex   = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
)

const postValidatePrompt = `You are reviewing an extracted drug-exception request. Judge whether it is internally consistent.
Specifically:
- Are all required fields present? (request_number, patient_name, patient_dob, member_id, prescriber, npi, drug, dosage, diagnosis, icd10, justification)
- Does the clinical justification reference or relate to the requested medication?
- Does the diagnosis plausibly match the requested medication's typical use?

Extracted request (JSON):
%s

Return JSON only.`

type PostValidate struct {
	Client *litertlm.Client
}

func (p *PostValidate) Name() store.Stage { return store.StagePostValidate }

func (p *PostValidate) Run(ctx context.Context, in workflow.StageInput) (workflow.StageOutput, error) {
	if in.Extracted == nil {
		return workflow.StageOutput{}, fmt.Errorf("post-validate: missing extracted payload")
	}

	missing := structuralCheck(in.Extracted)

	body, err := json.MarshalIndent(in.Extracted, "", "  ")
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("marshal extracted: %w", err)
	}

	result, err := litertlm.GenerateData[PostValidateResult](ctx, p.Client,
		fmt.Sprintf(postValidatePrompt, body),
	)
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("gemma post-validate: %w", err)
	}

	for _, m := range missing {
		if !contains(result.MissingFields, m) {
			result.MissingFields = append(result.MissingFields, m)
		}
	}
	if len(result.MissingFields) > 0 {
		result.Complete = false
	}

	if !result.Complete || !result.NarrativeMatchesDrug || !result.DiagnosisPlausible {
		notes := result.Reason
		if notes == "" {
			notes = fmt.Sprintf("incomplete or inconsistent (missing=%v narrative=%t diag=%t)",
				result.MissingFields, result.NarrativeMatchesDrug, result.DiagnosisPlausible)
		}
		return workflow.StageOutput{
			Outcome: workflow.OutcomeRouteReview,
			Lane:    store.LaneExtraction,
			Output:  result,
			Notes:   notes,
		}, nil
	}

	return workflow.StageOutput{
		Outcome: workflow.OutcomeAdvance,
		Output:  result,
		Notes:   "extraction looks consistent",
	}, nil
}

func structuralCheck(r *workflow.ExtractedRequest) []string {
	var missing []string
	required := map[string]string{
		"request_number": r.RequestNumber,
		"patient_name":   r.PatientName,
		"patient_dob":    r.PatientDOB,
		"member_id":      r.MemberID,
		"prescriber":     r.Prescriber,
		"npi":            r.NPI,
		"drug":           r.Drug,
		"dosage":         r.Dosage,
		"diagnosis":      r.Diagnosis,
		"icd10":          r.ICD10,
		"justification":  r.Justification,
	}
	for k, v := range required {
		if strings.TrimSpace(v) == "" {
			missing = append(missing, k)
		}
	}
	if r.NPI != "" && !npiRegex.MatchString(strings.TrimSpace(r.NPI)) {
		missing = append(missing, "npi (malformed: must be 10 digits)")
	}
	if r.ICD10 != "" && !icd10Regex.MatchString(strings.TrimSpace(strings.ToUpper(r.ICD10))) {
		missing = append(missing, "icd10 (malformed: expected like E11.9)")
	}
	if r.PatientDOB != "" && !dobRegex.MatchString(strings.TrimSpace(r.PatientDOB)) {
		missing = append(missing, "patient_dob (malformed: expected YYYY-MM-DD)")
	}
	return missing
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

var _ workflow.Stage = (*PostValidate)(nil)
