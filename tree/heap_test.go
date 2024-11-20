package tree

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMinHeap(t *testing.T) {
	type testCase struct {
		scenario string
		inputs   []uint32
		size     func(testCase) uint32
		expected []uint32
	}
	testCases := []testCase{
		{
			scenario: "add and pup",
			inputs:   []uint32{4, 3, 30, 6, 100, 1, 7, 11, 5},
			expected: []uint32{1, 3, 4, 5, 6, 7, 11, 30, 100},
			size:     func(tc testCase) uint32 { return uint32(len(tc.inputs)) },
		},
		{
			scenario: "size exceeded",
			inputs:   []uint32{4, 3, 30, 6, 100, 1, 7, 11, 5},
			expected: []uint32{1, 3, 4, 6, 30, 100},
			size:     func(tc testCase) uint32 { return uint32(len(tc.expected)) },
		},
	}
	for _, tc := range testCases {
		t.Run(tc.scenario, func(t *testing.T) {
			h := NewMinHeap(tc.size(tc))
			for _, input := range tc.inputs {
				h.Add(input)
			}
			for _, e := range tc.expected {
				assert.EqualValues(t, e, h.Pop())
			}

		})
	}
}
