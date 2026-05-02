package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

const systemPromptTemplate = `
You are a helpful weather assistant that ONLY handles weather requests
for US territories (the 50 states, DC, Puerto Rico, US Virgin Islands, Guam,
American Samoa, and Northern Mariana Islands). If the user asks about
a non-US location, politely let them know that this application only supports
weather forecasts for US territories. If the user asks something unrelated to
weather, politely let them know that this application only handles weather requests.
Do NOT call the get_weather tool for non-US locations. For valid US weather requests,
use the get_weather tool to fetch real forecast data, then provide a clear,
well-formatted summary.

## Weather condition output format
You must output the weather condition information as follows as outlined below:
- Format your response in **Markdown**
- Include all relevent weather information returned by the tool call
- Place sumamry detail and any additional advisory in the Detail bullet point
- For each weather conditions, use weather-related emojis to help convey conditions:
  * Use ☀️ - for sunny or clear day forcast
  * Use 🌤️ - for partial sunny condition
  * Use 🌦️ - for sunny followed by rainy conditions
  * Use ☁️ - for cloudy no rain conditions
  * Use 🌧️ - for mostly cloudy with possible rain conditions
  * Use ⛈️ - for cloudy, rainy, with possible thunderstorm conditions
  * Use 🌩️ - for cloudy, rainy, and lightning conditions
  * Use 🌨️ - for cloudy and snow conditions
  * Use 🌪️ - for possible tornadoes
  * Use ✨ - for clear night conditions
  * Use 💨 - for wind conditions
  * Use 🌡️ - for temperature conditions
  * Use ⏳ - for additional detail summary

Output layout as follows:

# <emoji representing average condition> Title

## <emoji for current condition> Current Conditions
* <emoji> <Today or Tonight>: sky condition (i.e. clear, partly cloudy, etc)
* <emoji> Temperature: <temp info>
* <emoji> Wind: <wind condition>
* <emoji> Detail: <condition summary or advisory>

## <calendar emoji> Upcoming Forecast

### <Day of week>: sky condition (i.e. clear, cloudy, etc) <emoji>
* <emoji> <Today or Tonight>: sky condition (i.e. clear, partly cloudy, etc)
* <emoji> Temperature: <temp info>
* <emoji> Wind: <wind condition>
* <emoji> Detail: <condition summary or advisory>

### <repeast ccondition for next day>

`

// getSystemPrompt returns the system prompt text (without tool definitions —
// those are rendered separately using native <|tool> tokens).
func getSystemPrompt() string {
	return strings.TrimSpace(systemPromptTemplate)
}

// renderToolDeclarations formats tool definitions in Gemma 4's native
// declaration syntax — the same shape the model's chat template (visible
// in the LiteRT-LM init log under "jinja_prompt_template") emits when
// `tools` are passed through the Conversation API. Example for our
// get_weather tool:
//
//	<|tool>declaration:get_weather{
//	  description:<|"|>...<|"|>,
//	  parameters:{
//	    properties:{location:{description:<|"|>...<|"|>,type:<|"|>STRING<|"|>}},
//	    required:[<|"|>location<|"|>],
//	    type:<|"|>OBJECT<|"|>
//	  }
//	}<tool|>
//
// We pass raw text (no chat template) via Session.GenerateContent, so the
// declaration text we emit must match the format the model was trained
// on. Feeding a JSON-blob declaration instead works on the larger E4B
// (which yzma-intro uses) but confuses the 2B model into emitting nested
// object args (e.g. `{location":{"location":"Miami, FL"}}`).
func renderToolDeclarations(tools []Tool) string {
	var b strings.Builder
	for _, t := range tools {
		b.WriteString("\n<|tool>declaration:")
		b.WriteString(t.Function.Name)
		b.WriteString("{description:")
		writeNativeString(&b, t.Function.Description)
		if len(t.Function.Parameters) > 0 {
			b.WriteString(",parameters:")
			writeNativeValue(&b, t.Function.Parameters)
		}
		b.WriteString("}<tool|>")
	}
	return b.String()
}

// writeNativeString writes a string in Gemma 4's <|"|>value<|"|> form.
func writeNativeString(b *strings.Builder, s string) {
	b.WriteString(`<|"|>`)
	b.WriteString(s)
	b.WriteString(`<|"|>`)
}

// writeNativeValue recursively renders a JSON-schema value in Gemma 4's
// native declaration syntax. The chat template's `dictsort` ensures
// alphabetical key order, which we reproduce so the rendered text is
// byte-identical to what the template produces for the same input.
//
// Handled value shapes:
//   - string                                 → <|"|>value<|"|> (uppercased for the "type" key)
//   - bool                                   → true | false
//   - []string / []interface{}               → [<|"|>v1<|"|>,<|"|>v2<|"|>]
//   - map[string]interface{}                 → {key:value,key:value}
//
// Number support is intentionally minimal — our schema doesn't use them.
func writeNativeValue(b *strings.Builder, v interface{}) {
	switch val := v.(type) {
	case string:
		writeNativeString(b, val)
	case bool:
		if val {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case []string:
		b.WriteString("[")
		for i, s := range val {
			if i > 0 {
				b.WriteString(",")
			}
			writeNativeString(b, s)
		}
		b.WriteString("]")
	case []interface{}:
		b.WriteString("[")
		for i, item := range val {
			if i > 0 {
				b.WriteString(",")
			}
			writeNativeValue(b, item)
		}
		b.WriteString("]")
	case map[string]interface{}:
		keys := make([]string, 0, len(val))
		for k := range val {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		b.WriteString("{")
		for i, k := range keys {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(k)
			b.WriteString(":")
			child := val[k]
			// Type keywords (STRING, OBJECT, ARRAY, ...) are uppercased
			// in the chat template's output. Mirror that so the model
			// sees text identical to its training distribution.
			if k == "type" {
				if s, ok := child.(string); ok {
					writeNativeString(b, strings.ToUpper(s))
					continue
				}
			}
			writeNativeValue(b, child)
		}
		b.WriteString("}")
	default:
		fmt.Fprintf(b, "%v", val)
	}
}

var (
	// Step 1: extract content between <|tool_call> and <tool_call|>.
	// (?s) enables dot-all so . matches newlines (long values may wrap).
	toolCallRe = regexp.MustCompile(`(?s)<\|tool_call>(.+?)<tool_call\|>`)
	// Step 2: extract tool name and JSON-like body from "call:name{...}".
	callBodyRe = regexp.MustCompile(`(?s)^call:(\w+)(\{.+\})$`)
	// Step 3a: rewrite `=` separator to `:` between an identifier and an
	// opening quote. Gemma 4 sometimes emits `{location="Miami"}` instead
	// of the documented `{location:<|"|>Miami<|"|>}` (observed empirically
	// in the spike). Normalising to `:` lets the same parser handle both.
	eqSepRe = regexp.MustCompile(`(\w+)\s*=\s*"`)
	// Step 3b: quote bare keys. The native format uses `key:` (unquoted)
	// where standard JSON wants `"key":`.
	bareKeyRe = regexp.MustCompile(`([{,])\s*(\w+)\s*:`)
)

// parseGemma4ToolCall parses a Gemma 4 native tool call from a model response.
// Three argument-encoding shapes have been observed and are all accepted:
//
//	<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>     // documented
//	<|tool_call>call:name{key="value"}<tool_call|>             // observed
//	<|tool_call>call:name{"key":"value"}<tool_call|>           // valid JSON
//
// Parsing proceeds in four steps:
//  1. Extract content from between <|tool_call> and <tool_call|>
//  2. Extract the tool name and JSON-like object from call:name{...}
//  3. Normalise the body to valid JSON: <|"|> → ", `=` → `:`, bare keys → "key"
//  4. Unmarshal the JSON into a map[string]string
func parseGemma4ToolCall(response string) (ToolCall, bool) {
	m := toolCallRe.FindStringSubmatch(response)
	if m == nil {
		return ToolCall{}, false
	}

	parts := callBodyRe.FindStringSubmatch(m[1])
	if parts == nil {
		return ToolCall{}, false
	}

	// Normalise to JSON. Order matters: replace <|"|> first so eqSepRe
	// can find a real opening quote, then rewrite `=` separators, then
	// quote any remaining bare keys, then escape raw newlines (which are
	// invalid inside JSON string literals).
	jsonStr := strings.ReplaceAll(parts[2], `<|"|>`, `"`)
	jsonStr = eqSepRe.ReplaceAllString(jsonStr, `$1:"`)
	jsonStr = bareKeyRe.ReplaceAllString(jsonStr, `$1"$2":`)
	jsonStr = strings.ReplaceAll(jsonStr, "\n", `\n`)

	var args map[string]string
	if err := json.Unmarshal([]byte(jsonStr), &args); err != nil {
		return ToolCall{}, false
	}

	return ToolCall{
		Type: "function",
		Function: ToolCallFunction{
			Name:      parts[1],
			Arguments: args,
		},
	}, true
}

// renderGemma4Prompt builds the initial prompt for the first generation pass.
//
// Gemma 4 format (https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4):
//
//	<|turn>system
//	[instructions]
//	<|tool>declaration:name{schema}<tool|><turn|>
//	<|turn>user
//	[question]<turn|>
//	<|turn>model
//
// LiteRT-LM's tokenizer recognises these literal control tokens as their
// special token IDs when the prompt is fed via Session.GenerateContent —
// no parseSpecial flag is needed. Verified by the spike at ../spike/.
func renderGemma4Prompt(systemPrompt, toolDeclarations, userPrompt string) string {
	var b strings.Builder
	b.WriteString("<|turn>system\n")
	b.WriteString(systemPrompt)
	b.WriteString(toolDeclarations)
	b.WriteString("<turn|>\n")
	b.WriteString("<|turn>user\n")
	b.WriteString(userPrompt)
	b.WriteString("<turn|>\n")
	b.WriteString("<|turn>model\n")
	return b.String()
}

// renderGemma4PromptWithToolResult builds the prompt for the second
// generation pass after a tool call. The model turn is left open so the
// model continues with the natural-language answer after the tool response.
//
//	<|turn>system
//	[instructions + tool declarations]<turn|>
//	<|turn>user
//	[question]<turn|>
//	<|turn>model
//	<|tool_call>call:name{args}<tool_call|>
//	<|tool_response>response:name{result}<tool_response|>
//	[model continues generating here]
func renderGemma4PromptWithToolResult(systemPrompt, toolDeclarations, userPrompt, toolCallText, toolResponseText string) string {
	var b strings.Builder
	b.WriteString("<|turn>system\n")
	b.WriteString(systemPrompt)
	b.WriteString(toolDeclarations)
	b.WriteString("<turn|>\n")
	b.WriteString("<|turn>user\n")
	b.WriteString(userPrompt)
	b.WriteString("<turn|>\n")
	b.WriteString("<|turn>model\n")
	b.WriteString(toolCallText)
	b.WriteString(toolResponseText)
	return b.String()
}
