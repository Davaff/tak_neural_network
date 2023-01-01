package tests

import (
	"context"
	"flag"
	"testing"

	"github.com/nelhage/taktician/ai"
	"github.com/nelhage/taktician/bitboard"
	"github.com/nelhage/taktician/ptn"
	"github.com/nelhage/taktician/tak"
)

var hashTests = flag.Bool("test-hash", false, "run hash collision tests")
var tps = flag.String("tps", "112S,12,1112S,x2/x2,121C,12S,x/1,21,2,2,2/x,2,1,1,1/2,x3,21 2 24", "run hash collision tests on tps")
var depth = flag.Int("depth", 5, "run hash collision tests to depth")

func wrapHash(tbl map[uint64][]*tak.Position, eval ai.EvaluationFunc) ai.EvaluationFunc {
	return func(c *bitboard.Constants, p *tak.Position) int64 {
		tbl[p.Hash()] = append(tbl[p.Hash()], p.Clone())
		return eval(c, p)
	}
}

func equal(a, b *tak.Position) bool {
	if a.ToMove() != b.ToMove() {
		return false
	}
	if a.White != b.White {
		return false
	}
	if a.Black != b.Black {
		return false
	}
	if a.Standing != b.Standing {
		return false
	}
	if a.Caps != b.Caps {
		return false
	}
	for i := range a.Height {
		if a.Height[i] != b.Height[i] {
			return false
		}
		if a.Stacks[i] != b.Stacks[i] {
			return false
		}
	}
	return true
}

func reportCollisions(t *testing.T, tbl map[uint64][]*tak.Position) {
	var n, collisions int
	for h, l := range tbl {
		n += len(l)
		p := l[0]
		for _, pp := range l[1:] {
			if !equal(p, pp) {
				t.Logf(" collision h=%x l=%q r=%q",
					h, ptn.FormatTPS(p), ptn.FormatTPS(pp),
				)
				collisions++
				break
			}
		}
	}

	t.Logf("evaluated %d positions and %d hashes, with %d collisions",
		n, len(tbl), collisions)
}

func TestHash(t *testing.T) {
	if !*hashTests {
		t.SkipNow()
	}
	testCollisions(t, tak.New(tak.Config{Size: 5}))
	p, e := ptn.ParseTPS(*tps)
	if e != nil {
		panic("bad tps")
	}
	testCollisions(t, p)
}

func testCollisions(t *testing.T, p *tak.Position) {
	tbl := make(map[uint64][]*tak.Position)
	ai := ai.NewMinimax(ai.MinimaxConfig{
		Size:     5,
		Depth:    *depth,
		Evaluate: wrapHash(tbl, ai.MakeEvaluator(5, nil)),
		TableMem: -1,
	})
	for i := 0; i < 4; i++ {
		m := ai.GetMove(context.Background(), p)
		p, _ = p.Move(m)
		if ok, _ := p.GameOver(); ok {
			break
		}
	}
	reportCollisions(t, tbl)
}
