package stages

import (
	"context"
	"fmt"

	"github.com/vladimirvivien/exception-workflow/internal/store"
	"github.com/vladimirvivien/exception-workflow/internal/workflow"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const extractPrompt = `Extract the fields below from this drug exception / prior-authorization request form.
Use exactly the labels printed on the form. Leave any field that is not visible or not filled in as an empty string.
Return JSON only.`

type Extract struct {
	Client *litertlm.Client
}

func (e *Extract) Name() store.Stage { return store.StageExtract }

func (e *Extract) Run(ctx context.Context, in workflow.StageInput) (workflow.StageOutput, error) {
	img, err := litertlm.ImageFromFile(in.ImagePath)
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("load image: %w", err)
	}

	extracted, err := litertlm.GenerateDataMulti[workflow.ExtractedRequest](ctx, e.Client,
		[]litertlm.Part{img, litertlm.Text(extractPrompt)},
	)
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("gemma extract: %w", err)
	}

	return workflow.StageOutput{
		Outcome:   workflow.OutcomeAdvance,
		Output:    extracted,
		Extracted: extracted,
		Notes:     fmt.Sprintf("extracted %d fields", countNonEmpty(extracted)),
	}, nil
}

func countNonEmpty(r *workflow.ExtractedRequest) int {
	n := 0
	for _, s := range []string{
		r.RequestNumber, r.PatientName, r.PatientDOB, r.MemberID,
		r.Prescriber, r.NPI, r.Drug, r.Dosage, r.Diagnosis, r.ICD10, r.Justification,
	} {
		if s != "" {
			n++
		}
	}
	return n
}

var _ workflow.Stage = (*Extract)(nil)
