package array

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBinarySearch(t *testing.T) {
	type testCase struct {
		scenario string

		l           []int
		val         int
		expected    int
		expectedErr error
	}
	testCases := []testCase{
		{
			scenario:    "success",
			l:           []int{1, 4, 6, 10, 11, 25, 28, 31, 37, 40, 44, 49, 50, 55, 60, 67, 69, 99},
			val:         55,
			expected:    13,
			expectedErr: nil,
		},
		{
			scenario:    "not found",
			l:           []int{1, 4, 6, 10, 11, 25, 28, 31, 37, 40, 44, 49, 50, 55, 60, 67, 69, 99},
			val:         100000,
			expected:    -1,
			expectedErr: errors.New("not found"),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.scenario, func(t *testing.T) {
			actual, err := BinarySearch(tc.l, tc.val)
			assert.Equal(t, tc.expectedErr, err)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestBubbleSort(t *testing.T) {
	var (
		l        = []int{3, 100, 1, 5, 63, 45, 47, 49, 91, 84, 85, 44, 2, 11}
		expected = []int{1, 2, 3, 5, 11, 44, 45, 47, 49, 63, 84, 85, 91, 100}
	)
	assert.EqualValues(t, expected, BubbleSort(l))
}

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
