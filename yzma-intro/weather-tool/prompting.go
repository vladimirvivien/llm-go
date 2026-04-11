package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/hybridgroup/yzma/pkg/message"
)

// renderGemmaPrompt builds a Gemma chat-format prompt from a chat history.
//
// Gemma's wire format (https://ai.google.dev/gemma/docs/core/prompt-structure):
//
//	<bos><start_of_turn>user
//	content<end_of_turn>
//	<start_of_turn>model
//	content<end_of_turn>
//	<start_of_turn>model        <- generation prompt
//
// The literal "<bos>", "<start_of_turn>", and "<end_of_turn>" markers in
// the output get converted to their special token IDs at tokenization
// time via parseSpecial=true in llama.Tokenize. The same format works
// for every Gemma 4 variant (E2B, E4B, all quantizations) because the
// Gemma family has used the same turn markers since Gemma 2.
//
// Two Gemma-specific quirks the renderer handles:
//
//  1. Gemma was trained with only "user" and "model" roles. There is no
//     "system" role — the official chat template explicitly raises an
//     exception if it sees one. We buffer system message content here
//     and merge it into the FIRST user message that follows, separated
//     by a blank line.
//  2. yzma uses "assistant" as the role label for model output messages.
//     We rename it to "model" before writing the turn block.
//
// Tool calling: this app teaches the model a manual <tool_call>{...}</tool_call>
// XML convention via the system prompt. message.Tool entries (assistant
// turns containing one or more tool calls) are rendered as model turns
// whose content is the same XML block the model was taught to emit.
// message.ToolResponse entries are rendered as user turns prefixed with
// "Tool result for <name>:". This keeps the conversation self-consistent:
// the model sees its own previous tool calls and the tool outputs in
// exactly the syntax it produces.
func renderGemmaPrompt(messages []message.Message) string {
	var b strings.Builder
	b.WriteString("<bos>")

	// Buffer for system message content that we'll prepend to the next
	// user turn. Gemma has no system role; the convention is to merge
	// system instructions into the first user message of the conversation.
	var systemPrefix strings.Builder

	// flushOrphanSystem writes any buffered system content as a standalone
	// user turn. Called before any non-user, non-system turn so that
	// system content always appears in the conversation BEFORE the turn
	// that follows it. The "merge into user turn" optimization in the
	// user case below is preferred when possible (it produces fewer total
	// turns), but if a model/tool/tool-response turn comes before any
	// user turn, this is the fallback that keeps the system content from
	// being silently dropped or written in the wrong position.
	flushOrphanSystem := func() {
		if systemPrefix.Len() == 0 {
			return
		}
		b.WriteString("<start_of_turn>user\n")
		b.WriteString(systemPrefix.String())
		b.WriteString("<end_of_turn>\n")
		systemPrefix.Reset()
	}

	for _, m := range messages {
		switch msg := m.(type) {
		case message.Chat:
			switch msg.Role {
			case "system":
				if systemPrefix.Len() > 0 {
					systemPrefix.WriteString("\n\n")
				}
				systemPrefix.WriteString(msg.Content)
			case "user":
				// Merge any buffered system content into this user turn.
				b.WriteString("<start_of_turn>user\n")
				if systemPrefix.Len() > 0 {
					b.WriteString(systemPrefix.String())
					b.WriteString("\n\n")
					systemPrefix.Reset()
				}
				b.WriteString(msg.Content)
				b.WriteString("<end_of_turn>\n")
			case "assistant", "model":
				flushOrphanSystem()
				b.WriteString("<start_of_turn>model\n")
				b.WriteString(msg.Content)
				b.WriteString("<end_of_turn>\n")
			}

		case message.Tool:
			// An assistant turn containing one or more tool calls. Encode
			// each call as a <tool_call>{...}</tool_call> block matching
			// the format the system prompt teaches the model to emit, and
			// write the combined block as a single model turn.
			flushOrphanSystem()
			b.WriteString("<start_of_turn>model\n")
			for i, tc := range msg.ToolCalls {
				if i > 0 {
					b.WriteByte('\n')
				}
				argsJSON, _ := json.Marshal(tc.Function.Arguments)
				fmt.Fprintf(
					&b, `<tool_call>{"name": %q, "arguments": %s}</tool_call>`,
					tc.Function.Name, string(argsJSON),
				)
			}
			b.WriteString("<end_of_turn>\n")

		case message.ToolResponse:
			// Result returned from executing a tool. Encode as a user
			// turn so the model sees the result in the next iteration.
			flushOrphanSystem()
			b.WriteString("<start_of_turn>user\n")
			fmt.Fprintf(&b, `Tool result for %s: %s`, msg.Name, msg.Content)
			b.WriteString("<end_of_turn>\n")
		}
	}

	// Edge case: trailing system content with no following message.
	flushOrphanSystem()

	// Generation prompt — tells the model it's its turn to speak.
	b.WriteString("<start_of_turn>model\n")
	return b.String()
}
