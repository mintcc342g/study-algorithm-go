package set

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSetAdd(t *testing.T) {
	type testCase struct {
		scenario string
		set      *IntSet
		value    int
		expected error
	}
	testCases := []testCase{
		{
			scenario: "success",
			set: &IntSet{
				elements: []int{},
				index:    []bool{},
			},
			value:    1,
			expected: nil,
		},
		{
			scenario: "duplicated element",
			set: &IntSet{
				elements: []int{0, 1},
				index:    []bool{true, true},
			},
			value:    1,
			expected: errors.New("invalid value or already exist"),
		},
		{
			scenario: "out of range",
			set: &IntSet{
				elements: []int{},
				index:    []bool{},
			},
			value:    -1,
			expected: errors.New("invalid value or already exist"),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.scenario, func(t *testing.T) {
			actualErr := tc.set.Add(tc.value)
			assert.Equal(t, tc.expected, actualErr)
		})
	}
}

func TestSetRemove(t *testing.T) {
	type testCase struct {
		scenario string
		set      *IntSet
		value    int
		expected error
	}
	testCases := []testCase{
		{
			scenario: "success",
			set: &IntSet{
				elements: []int{0, 1},
				index:    []bool{true, true},
			},
			value:    1,
			expected: nil,
		},
		{
			scenario: "not exist element",
			set: &IntSet{
				elements: []int{0, 1},
				index:    []bool{true, true},
			},
			value:    2,
			expected: errors.New("invalid value or not exist"),
		},
		{
			scenario: "out of range",
			set: &IntSet{
				elements: []int{},
				index:    []bool{},
			},
			value:    -1,
			expected: errors.New("invalid value or not exist"),
		},
		{
			scenario: "fatal bug",
			set: &IntSet{
				elements: []int{0, 0},
				index:    []bool{true, true},
			},
			value:    1,
			expected: errors.New("[FATAL] fail to remove the value in elements; should not be reach here"),
		},
	}
	for _, tc := range testCases {
		t.Run(tc.scenario, func(t *testing.T) {
			actualErr := tc.set.Remove(tc.value)
			assert.Equal(t, tc.expected, actualErr)
		})
	}
}

func TestSetContains(t *testing.T) {
	type testCase struct {
		scenario string
		set      *IntSet
		value    int
		expected bool
	}
	testCases := []testCase{
		{
			scenario: "exist element",
			set: &IntSet{
				elements: []int{0, 1},
				index:    []bool{true, true},
			},
			value:    1,
			expected: true,
		},
		{
			scenario: "not exist element",
			set: &IntSet{
				elements: []int{0, 1},
				index:    []bool{true, true},
			},
			value:    2,
			expected: false,
		},
		{
			scenario: "out of range",
			set: &IntSet{
				elements: []int{},
				index:    []bool{},
			},
			value:    -1,
			expected: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.scenario, func(t *testing.T) {
			actualErr := tc.set.Contains(tc.value)
			assert.Equal(t, tc.expected, actualErr)
		})
	}
}
