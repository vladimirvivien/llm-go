package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

type FormSpec struct {
	Name          string
	Notes         string
	Patient       string
	DOB           string
	MemberID      string
	RequestNumber string
	Prescriber    string
	NPI           string
	Drug          string
	Dosage        string
	Diagnosis     string
	ICD10         string
	Justification string
	Blank         bool
	NotAForm      bool
}

func defaultForms() []FormSpec {
	return []FormSpec{
		{
			Name:          "01-clean.png",
			Notes:         "complete + on-formulary -> auto-approve",
			Patient:       "Maria Lopez",
			DOB:           "1972-04-18",
			MemberID:      "M-882-441-99",
			RequestNumber: "PA-2026-00141",
			Prescriber:    "Dr. Anita Shah, MD",
			NPI:           "1396745812",
			Drug:          "Metformin",
			Dosage:        "500 mg, twice daily",
			Diagnosis:     "Type 2 diabetes mellitus",
			ICD10:         "E11.9",
			Justification: "Patient newly diagnosed with type 2 diabetes; A1c 8.4. First-line therapy per guideline.",
		},
		{
			Name:          "02-missing-npi.png",
			Notes:         "missing prescriber NPI -> human-review (extraction)",
			Patient:       "James Carter",
			DOB:           "1965-09-02",
			MemberID:      "M-217-009-43",
			RequestNumber: "PA-2026-00142",
			Prescriber:    "Dr. Robert Liu, MD",
			NPI:           "",
			Drug:          "Atorvastatin",
			Dosage:        "40 mg nightly",
			Diagnosis:     "Hyperlipidemia",
			ICD10:         "E78.5",
			Justification: "LDL 188 despite diet; statin indicated.",
		},
		{
			Name:          "03-off-formulary.png",
			Notes:         "off-formulary specialty drug -> human-review (decision)",
			Patient:       "Priya Natarajan",
			DOB:           "1981-12-30",
			MemberID:      "M-554-330-11",
			RequestNumber: "PA-2026-00143",
			Prescriber:    "Dr. Helen Park, MD",
			NPI:           "1740029931",
			Drug:          "Adalimumab",
			Dosage:        "40 mg subcutaneous, every 2 weeks",
			Diagnosis:     "Rheumatoid arthritis",
			ICD10:         "M06.9",
			Justification: "Severe RA; failed methotrexate after 6 months. Specialty biologic requested.",
		},
		{
			Name:  "04-blank.png",
			Notes: "blank form -> rejected at pre-validate",
			Blank: true,
		},
		{
			Name:     "05-not-a-form.png",
			Notes:    "not a drug-exception form -> rejected at pre-validate",
			NotAForm: true,
		},
	}
}

const (
	width  = 1100
	height = 1500
)

func main() {
	out := flag.String("out", "testdata/forms", "output directory")
	flag.Parse()

	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	for _, spec := range defaultForms() {
		path := filepath.Join(*out, spec.Name)
		if err := writeForm(path, spec); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", spec.Name, err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s  (%s)\n", path, spec.Notes)
	}
}

func writeForm(path string, spec FormSpec) error {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	fillRect(img, img.Bounds(), color.White)

	if spec.NotAForm {
		drawNotAForm(img)
	} else if spec.Blank {
		drawBlankFrame(img)
	} else {
		drawForm(img, spec)
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	return png.Encode(f, img)
}

func fillRect(img *image.RGBA, r image.Rectangle, c color.Color) {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			img.Set(x, y, c)
		}
	}
}

func drawText(img *image.RGBA, x, y int, s string, c color.Color) {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(x, y),
	}
	d.DrawString(s)
}

func drawTextScaled(img *image.RGBA, x, y, scale int, s string, c color.Color) {
	face := basicfont.Face7x13
	src := image.NewUniform(c)

	for cx, ch := range s {
		drawScaledGlyph(img, x+cx*7*scale, y, scale, ch, face, src)
	}
}

func drawScaledGlyph(img *image.RGBA, x, y, scale int, ch rune, face *basicfont.Face, src image.Image) {
	tmp := image.NewRGBA(image.Rect(0, 0, 7, 13))
	d := &font.Drawer{
		Dst:  tmp,
		Src:  src,
		Face: face,
		Dot:  fixed.P(0, 11),
	}
	d.DrawString(string(ch))

	for sy := 0; sy < 13; sy++ {
		for sx := 0; sx < 7; sx++ {
			c := tmp.RGBAAt(sx, sy)
			if c.A == 0 {
				continue
			}
			for dy := 0; dy < scale; dy++ {
				for dx := 0; dx < scale; dx++ {
					img.Set(x+sx*scale+dx, y-(13-sy)*scale+dy, c)
				}
			}
		}
	}
}

func drawHLine(img *image.RGBA, x1, y, x2 int, c color.Color) {
	for x := x1; x <= x2; x++ {
		img.Set(x, y, c)
	}
}

func drawVLine(img *image.RGBA, x, y1, y2 int, c color.Color) {
	for y := y1; y <= y2; y++ {
		img.Set(x, y, c)
	}
}

func drawBox(img *image.RGBA, r image.Rectangle, c color.Color) {
	drawHLine(img, r.Min.X, r.Min.Y, r.Max.X, c)
	drawHLine(img, r.Min.X, r.Max.Y, r.Max.X, c)
	drawVLine(img, r.Min.X, r.Min.Y, r.Max.Y, c)
	drawVLine(img, r.Max.X, r.Min.Y, r.Max.Y, c)
}

func drawBlankFrame(img *image.RGBA) {
	black := color.Black
	drawBox(img, image.Rect(40, 40, width-40, height-40), black)
}

func drawNotAForm(img *image.RGBA) {
	black := color.Black
	drawTextScaled(img, 80, 200, 4, "MEETING NOTES", black)
	lines := []string{
		"Q2 budget review - moved to next Tuesday.",
		"Action items from last week:",
		"  - revisit vendor short-list",
		"  - circulate updated org chart",
		"  - schedule offsite for July",
		"",
		"Open questions: hiring freeze status?",
	}
	for i, line := range lines {
		drawText(img, 80, 280+i*30, line, black)
	}
}

func drawForm(img *image.RGBA, spec FormSpec) {
	black := color.Black
	gray := color.RGBA{0x55, 0x55, 0x55, 0xff}

	drawBox(img, image.Rect(40, 40, width-40, height-40), black)

	drawTextScaled(img, 80, 110, 3, "DRUG EXCEPTION REQUEST", black)
	drawText(img, 80, 145, "Prior Authorization / Formulary Exception", gray)
	drawHLine(img, 80, 165, width-80, black)

	row := func(y int, label, value string) {
		drawText(img, 80, y, label, gray)
		drawText(img, 360, y, value, black)
		drawHLine(img, 360, y+8, width-80, gray)
	}

	row(220, "Request Number:", spec.RequestNumber)
	row(265, "Patient Name:", spec.Patient)
	row(310, "Date of Birth:", spec.DOB)
	row(355, "Member / Patient ID:", spec.MemberID)

	drawTextScaled(img, 80, 430, 2, "Prescriber", black)
	drawHLine(img, 80, 450, width-80, gray)
	row(495, "Prescriber Name:", spec.Prescriber)
	row(540, "NPI:", spec.NPI)

	drawTextScaled(img, 80, 620, 2, "Medication", black)
	drawHLine(img, 80, 640, width-80, gray)
	row(685, "Drug Requested:", spec.Drug)
	row(730, "Dosage:", spec.Dosage)
	row(775, "Diagnosis:", spec.Diagnosis)
	row(820, "ICD-10 Code:", spec.ICD10)

	drawTextScaled(img, 80, 900, 2, "Clinical Justification", black)
	drawHLine(img, 80, 920, width-80, gray)
	wrapAndDrawText(img, 80, 960, width-160, spec.Justification, black)

	drawText(img, 80, height-90, "Prescriber signature: ____________________________", gray)
	drawText(img, 80, height-65, fmt.Sprintf("Submitted via PA Portal v3.4 / form #%s", spec.RequestNumber), gray)
}

func wrapAndDrawText(img *image.RGBA, x, y, maxWidth int, s string, c color.Color) {
	if s == "" {
		return
	}
	const charW = 7
	maxChars := maxWidth / charW
	words := strings.Fields(s)
	var line string
	yy := y
	for _, w := range words {
		probe := line
		if probe != "" {
			probe += " "
		}
		probe += w
		if len(probe) > maxChars && line != "" {
			drawText(img, x, yy, line, c)
			yy += 22
			line = w
			continue
		}
		line = probe
	}
	if line != "" {
		drawText(img, x, yy, line, c)
	}
}
