package providers

import (
	"context"
	"encoding/json"
	"io"
	"regexp"
	"time"

	"go.uber.org/zap"
)

// Gemini Web now streams most chat replies over a separate WebChannel session
// (geminiweb-pa.clients6.google.com/v1/processSession). StreamGenerate still
// accepts the turn and Gemini still writes the finished reply into the
// conversation, but for many models the StreamGenerate response itself only
// carries the "turn accepted" frame and then idles until the client gives up.
//
// readGenerateStream therefore reads the stream incrementally. When it goes
// quiet after the turn was accepted, the conversation is polled (hNvQHb) and
// the stored reply for that exact turn is returned in StreamGenerate shape, so
// parseResponse handles both paths identically.
const (
	streamIdleBeforePoll   = 5 * time.Second
	conversationPollPeriod = 3 * time.Second
)

// acceptedTurnRegex finds the conversation and response IDs inside the
// JSON-escaped payload of the first StreamGenerate frame.
var acceptedTurnRegex = regexp.MustCompile(`\\"(c_[0-9A-Za-z]+)\\",\\"(r_[0-9A-Za-z]+)\\"`)

func (c *Client) readGenerateStream(ctx context.Context, body io.ReadCloser) ([]byte, error) {
	type chunk struct {
		data []byte
		err  error
	}
	chunks := make(chan chunk, 16)
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		buf := make([]byte, 32*1024)
		for {
			n, err := body.Read(buf)
			if n > 0 {
				select {
				case chunks <- chunk{data: append([]byte(nil), buf[:n]...)}:
				case <-stop:
					return
				}
			}
			if err != nil {
				select {
				case chunks <- chunk{err: err}:
				case <-stop:
				}
				return
			}
		}
	}()

	ticker := time.NewTicker(conversationPollPeriod)
	defer ticker.Stop()
	var received []byte
	var conversationID, responseID string
	lastData := time.Now()
	for {
		select {
		case <-ctx.Done():
			_ = body.Close()
			return received, ctx.Err()
		case ch := <-chunks:
			if ch.err != nil {
				if ch.err == io.EOF {
					return received, nil
				}
				return received, ch.err
			}
			received = append(received, ch.data...)
			lastData = time.Now()
			if responseID == "" {
				if m := acceptedTurnRegex.FindSubmatch(received); m != nil {
					conversationID, responseID = string(m[1]), string(m[2])
					c.log.Debug("Gemini accepted the turn",
						zap.String("conversation_id", conversationID),
						zap.String("response_id", responseID))
				}
			}
		case <-ticker.C:
			if responseID == "" || time.Since(lastData) < streamIdleBeforePoll {
				continue
			}
			reply, err := c.storedReply(ctx, conversationID, responseID)
			if err != nil {
				c.log.Debug("Conversation poll failed", zap.String("conversation_id", conversationID), zap.Error(err))
				continue
			}
			if reply == nil {
				c.log.Debug("Reply not stored in the conversation yet",
					zap.String("conversation_id", conversationID),
					zap.Duration("stream_idle", time.Since(lastData)))
				continue
			}
			c.log.Info("StreamGenerate went quiet; returning the reply stored in the conversation",
				zap.String("conversation_id", conversationID),
				zap.String("response_id", responseID),
				zap.Duration("stream_idle", time.Since(lastData)))
			// Unblocks the reader goroutine; the caller's Close is then a no-op.
			_ = body.Close()
			return reply, nil
		}
	}
}

// storedReply returns the finished reply for responseID, encoded as a
// StreamGenerate frame, or nil when the conversation does not hold it yet.
func (c *Client) storedReply(ctx context.Context, conversationID, responseID string) ([]byte, error) {
	body, err := c.readConversationBody(ctx, conversationID)
	if err != nil {
		return nil, err
	}
	candidates := conversationTurnCandidates(body, responseID)
	if !finishedCandidates(candidates) {
		return nil, nil
	}
	payload, err := json.Marshal([]interface{}{nil, []interface{}{conversationID, responseID}, nil, nil, candidates})
	if err != nil {
		return nil, err
	}
	return json.Marshal([]interface{}{[]interface{}{"wrb.fr", nil, string(payload)}})
}

// conversationTurnCandidates returns the reply candidates of the turn whose
// response ID is responseID. Turns are [[cid,rid], null, userMessage,
// [candidates, …], …].
func conversationTurnCandidates(body []byte, responseID string) []interface{} {
	for _, rawTurn := range conversationTurns(body) {
		turn, _ := rawTurn.([]interface{})
		if len(turn) < 4 {
			continue
		}
		ids, _ := turn[0].([]interface{})
		if len(ids) < 2 {
			continue
		}
		if rid, _ := ids[1].(string); rid != responseID {
			continue
		}
		replies, _ := turn[3].([]interface{})
		if len(replies) == 0 {
			return nil
		}
		candidates, _ := replies[0].([]interface{})
		return candidates
	}
	return nil
}

// finishedCandidates reports whether the first candidate holds reply text and
// is not marked as still streaming. Field 8 is [1] while Gemini is writing and
// [2] once the candidate is complete; it may be absent in stored turns.
func finishedCandidates(candidates []interface{}) bool {
	if len(candidates) == 0 {
		return false
	}
	candidate, _ := candidates[0].([]interface{})
	if len(candidate) < 2 {
		return false
	}
	parts, _ := candidate[1].([]interface{})
	if len(parts) == 0 {
		return false
	}
	if text, _ := parts[0].(string); text == "" {
		return false
	}
	if len(candidate) > 8 {
		if status, ok := candidate[8].([]interface{}); ok && len(status) > 0 {
			if code, ok := status[0].(float64); ok && code == 1 {
				return false
			}
		}
	}
	return true
}
