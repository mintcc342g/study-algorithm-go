package array

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQuickSort(t *testing.T) {
	type testCase struct {
		scenario string

		l        []int
		desc     bool
		expected []int
	}
	testCases := []testCase{
		{
			scenario: "asc",
			l:        []int{2, 7, 4, 3, 8},
			desc:     false,
			expected: []int{2, 3, 4, 7, 8},
		},
		{
			scenario: "desc",
			l:        []int{2, 7, 4, 3, 8},
			desc:     true,
			expected: []int{8, 7, 4, 3, 2},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.scenario, func(t *testing.T) {
			assert.EqualValues(t, tc.expected, QuickSort(tc.l, tc.desc))
		})
	}
}

func TestQuickSelection(t *testing.T) {
	type testCase struct {
		scenario string

		l        []int
		k        uint
		largest  bool
		expected int
	}
	testCases := []testCase{
		{
			scenario: "smallest",
			l:        []int{332, 5, 3, 10, 100, 21, 82, 4, 37, 99, 7},
			k:        3,
			largest:  false,
			expected: 5,
		},
		{
			scenario: "largest",
			l:        []int{332, 5, 3, 10, 100, 21, 82, 4, 37, 99, 7},
			k:        3,
			largest:  true,
			expected: 99,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.scenario, func(t *testing.T) {
			assert.EqualValues(t, tc.expected, QuickSelection(tc.l, tc.k, tc.largest))
		})
	}
}
