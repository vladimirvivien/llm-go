package stages

import (
	"context"
	"fmt"
	"os"

	"github.com/vladimirvivien/exception-workflow/internal/store"
	"github.com/vladimirvivien/exception-workflow/internal/workflow"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type PreValidateResult struct {
	IsForm           bool   `json:"is_form" description:"Is this image a drug-exception / prior-authorization request form?"`
	HasPatient       bool   `json:"has_patient" description:"Does the form have a patient name filled in?"`
	HasRequestNumber bool   `json:"has_request_number" description:"Does the form have a request or PA number filled in?"`
	HasMedication    bool   `json:"has_medication" description:"Does the form name a requested medication?"`
	Notes            string `json:"notes" description:"One short sentence describing what the form looks like"`
}

const preValidatePrompt = `You are a triage assistant for a prior-authorization workflow.
Look at this scanned form image and answer four yes/no questions:
1) Is this a drug exception / prior authorization request form?
2) Does it have a patient name filled in?
3) Does it have a request number or PA number filled in?
4) Does it name a requested medication?
Then add one short sentence describing what the form looks like.
Return JSON only.`

type PreValidate struct {
	Client *litertlm.Client
}

func (p *PreValidate) Name() store.Stage { return store.StagePreValidate }

func (p *PreValidate) Run(ctx context.Context, in workflow.StageInput) (workflow.StageOutput, error) {
	info, err := os.Stat(in.ImagePath)
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("stat image: %w", err)
	}
	if info.Size() < 1024 {
		return workflow.StageOutput{
			Outcome: workflow.OutcomeReject,
			Notes:   "image too small to be a form",
			Output:  map[string]any{"reason": "size", "bytes": info.Size()},
		}, nil
	}

	img, err := litertlm.ImageFromFile(in.ImagePath)
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("load image: %w", err)
	}

	result, err := litertlm.GenerateDataMulti[PreValidateResult](ctx, p.Client,
		[]litertlm.Part{img, litertlm.Text(preValidatePrompt)},
	)
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("gemma pre-validate: %w", err)
	}

	if !result.IsForm {
		return workflow.StageOutput{
			Outcome: workflow.OutcomeReject,
			Notes:   "image is not a drug-exception form",
			Output:  result,
		}, nil
	}

	missing := []string{}
	if !result.HasPatient {
		missing = append(missing, "patient name")
	}
	if !result.HasRequestNumber {
		missing = append(missing, "request number")
	}
	if !result.HasMedication {
		missing = append(missing, "medication")
	}
	if len(missing) > 0 {
		return workflow.StageOutput{
			Outcome: workflow.OutcomeReject,
			Notes:   fmt.Sprintf("form missing required markers: %v", missing),
			Output:  result,
		}, nil
	}

	return workflow.StageOutput{
		Outcome: workflow.OutcomeAdvance,
		Output:  result,
		Notes:   "structural markers present",
	}, nil
}

var _ workflow.Stage = (*PreValidate)(nil)
