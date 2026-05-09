package stages

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/vladimirvivien/exception-workflow/internal/store"
	"github.com/vladimirvivien/exception-workflow/internal/workflow"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type DecideResult struct {
	Score     int    `json:"score"`
	Decision  string `json:"decision"`
	Rationale string `json:"rationale"`
	RawText   string `json:"raw_text,omitempty"`
}

type formularyIn struct {
	Drug string `json:"drug" description:"Generic or brand drug name to look up"`
}

type formularyOut struct {
	DrugName    string   `json:"drug_name"`
	Found       bool     `json:"found"`
	OnFormulary bool     `json:"on_formulary"`
	RequiresPA  bool     `json:"requires_pa"`
	StepTherapy []string `json:"step_therapy"`
}

const decideSystemPrompt = `You are a prior-authorization triage assistant. You make an initial assessment of a drug-exception request based on institutional rules.

Decision rules:
- If the requested drug is on formulary AND does NOT require prior authorization, score 100.
- If the drug is on formulary AND requires prior authorization AND the patient has tried all step-therapy prerequisites mentioned in the justification, score 100.
- If the drug is on formulary AND requires PA but the justification does not mention all step-therapy prerequisites, score 60.
- If the drug is OFF formulary, score at most 40 — even if the justification is strong.
- If you cannot find the drug in the formulary, score 50.

ALWAYS call the lookup_formulary tool to check the requested drug before scoring.
Return your final answer in this exact format on its own line:
SCORE: <0-100>
DECISION: <approved|denied|needs-review>
RATIONALE: <one or two sentences>`

const decideUserPromptTmpl = `Drug-exception request to assess:

%s

Look up the drug in the formulary, then return SCORE / DECISION / RATIONALE.`

type Decide struct {
	Client           *litertlm.Client
	Store            *store.Store
	AutoApproveScore int
}

func (d *Decide) Name() store.Stage { return store.StageDecide }

func (d *Decide) Run(ctx context.Context, in workflow.StageInput) (workflow.StageOutput, error) {
	if in.Extracted == nil {
		return workflow.StageOutput{}, fmt.Errorf("decide: missing extracted payload")
	}

	tool, err := litertlm.RegisterTool(d.Client, "lookup_formulary",
		"Look up a drug's formulary status, prior-authorization requirement, and step-therapy prerequisites.",
		func(ctx context.Context, in formularyIn) (formularyOut, error) {
			drug := strings.ToLower(strings.TrimSpace(in.Drug))
			if drug == "" {
				return formularyOut{Found: false}, fmt.Errorf("drug is required")
			}
			entry, err := d.Store.GetFormularyEntry(ctx, drug)
			if err != nil {
				return formularyOut{DrugName: drug, Found: false}, nil
			}
			return formularyOut{
				DrugName:    entry.DrugName,
				Found:       true,
				OnFormulary: entry.OnFormulary,
				RequiresPA:  entry.RequiresPA,
				StepTherapy: entry.StepTherapy,
			}, nil
		},
		litertlm.WithToolPolicy(litertlm.ToolPolicyInformOnError),
	)
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("register lookup tool: %w", err)
	}

	chat, err := d.Client.NewChat(ctx,
		litertlm.WithSystemPrompt(decideSystemPrompt),
		litertlm.WithTool(tool),
		litertlm.WithMaxToolHops(4),
	)
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("new chat: %w", err)
	}
	defer func() { _ = chat.Close() }()

	body, err := json.MarshalIndent(in.Extracted, "", "  ")
	if err != nil {
		return workflow.StageOutput{}, fmt.Errorf("marshal extracted: %w", err)
	}

	reply, err := chat.Send(ctx, fmt.Sprintf(decideUserPromptTmpl, body))
	if err != nil {
		var hops *litertlm.ToolHopsError
		if errors.As(err, &hops) {
			return workflow.StageOutput{
				Outcome: workflow.OutcomeRouteReview,
				Lane:    store.LaneDecision,
				Notes:   fmt.Sprintf("tool hops exceeded after %d iterations", hops.Hops),
				Output:  DecideResult{Score: 0, Decision: "needs-review", Rationale: "tool hops exceeded"},
				Raw:     hops.LastReply.Raw(),
			}, nil
		}
		return workflow.StageOutput{}, fmt.Errorf("chat.send: %w", err)
	}

	parsed := parseDecideReply(reply.Text())
	parsed.RawText = reply.Text()

	cutoff := d.AutoApproveScore
	if cutoff <= 0 {
		cutoff = 100
	}
	if parsed.Score >= cutoff {
		return workflow.StageOutput{
			Outcome: workflow.OutcomeApprove,
			Score:   parsed.Score,
			Output:  parsed,
			Raw:     reply.Raw(),
			Notes:   parsed.Rationale,
		}, nil
	}

	return workflow.StageOutput{
		Outcome: workflow.OutcomeRouteReview,
		Lane:    store.LaneDecision,
		Score:   parsed.Score,
		Output:  parsed,
		Raw:     reply.Raw(),
		Notes:   parsed.Rationale,
	}, nil
}

var (
	scoreRe     = regexp.MustCompile(`(?i)\bSCORE:\s*(\d{1,3})`)
	decisionRe  = regexp.MustCompile(`(?i)\bDECISION:\s*([a-z\-]+)`)
	rationaleRe = regexp.MustCompile(`(?is)\bRATIONALE:\s*(.+?)(?:\n[A-Z]+:|$)`)
)

func parseDecideReply(text string) DecideResult {
	r := DecideResult{}
	if m := scoreRe.FindStringSubmatch(text); len(m) == 2 {
		if n, err := strconv.Atoi(m[1]); err == nil {
			if n < 0 {
				n = 0
			} else if n > 100 {
				n = 100
			}
			r.Score = n
		}
	}
	if m := decisionRe.FindStringSubmatch(text); len(m) == 2 {
		r.Decision = strings.ToLower(strings.TrimSpace(m[1]))
	}
	if m := rationaleRe.FindStringSubmatch(text); len(m) == 2 {
		r.Rationale = strings.TrimSpace(m[1])
	}
	return r
}

var _ workflow.Stage = (*Decide)(nil)
