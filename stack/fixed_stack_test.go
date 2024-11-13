package stack

import (
	"fmt"
	"study-algorithm-go/utils"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStackLen(t *testing.T) {
	var (
		size     = utils.RandInt(5)
		item     = utils.RandInt(5)
		expected = 1
	)

	fixedStk := NewFixedStack(size)
	fixedStk.Push(item)

	assert.Equal(t, expected, fixedStk.Len())
}

func TestStackPush(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				size     = utils.RandInt(5)
				expected = utils.RandInt(10)
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(expected)
			assert.NoError(t, err)

			actualVal, err := fixedStk.Pop()
			assert.NoError(t, err)
			assert.EqualValues(t, expected, actualVal)
		},
		"fail to push": func(t *testing.T) {
			var (
				size        = 0
				expectedErr = "capacity over"
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(utils.RandInt(10))
			assert.ErrorContains(t, err, expectedErr)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestStackPop(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				size     = 1
				expected = utils.RandInt(10)
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(expected)
			assert.NoError(t, err)

			actual, err := fixedStk.Peek()
			assert.NoError(t, err)
			assert.EqualValues(t, expected, actual)
		},
		"fail to pop": func(t *testing.T) {
			var (
				size        = 5
				expectedVal = 0
				expectedErr = "empty stack"
			)

			fixedStk := NewFixedStack(size)

			actualVal, err := fixedStk.Pop()
			assert.ErrorContains(t, err, expectedErr)
			assert.EqualValues(t, expectedVal, actualVal)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestStackPeek(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				size     = 5
				expected = 10
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(expected)
			assert.NoError(t, err)

			actual, err := fixedStk.Peek()
			assert.NoError(t, err)
			assert.EqualValues(t, expected, actual)
		},
		"empty stack": func(t *testing.T) {
			var (
				size        = 5
				expectedVal = 0
				expectedErr = "empty stack"
			)

			fixedStk := NewFixedStack(size)
			actualVal, err := fixedStk.Pop()
			assert.ErrorContains(t, err, expectedErr)
			assert.EqualValues(t, expectedVal, actualVal)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestStackClear(t *testing.T) {
	var (
		size     = utils.RandInt(5)
		val      = utils.RandInt(5)
		expected = 0
	)

	fixedStk := NewFixedStack(size)
	fixedStk.Push(val)
	fixedStk.Clear()

	assert.EqualValues(t, expected, fixedStk.Len())
}

func TestStackFind(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				size     = 5
				val      = 10
				expected = 1
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(utils.RandInt(9))
			assert.NoError(t, err)
			err = fixedStk.Push(val)
			assert.NoError(t, err)

			actual, err := fixedStk.Find(val)
			assert.NoError(t, err)
			assert.EqualValues(t, expected, actual)
		},
		"fail to find": func(t *testing.T) {
			var (
				size        = 5
				val         = utils.RandInt(10) + 10
				expectedVal = 0
				expectedErr = "not found"
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(utils.RandInt(10))
			assert.NoError(t, err)

			actualVal, err := fixedStk.Find(val)
			assert.ErrorContains(t, err, expectedErr)
			assert.EqualValues(t, expectedVal, actualVal)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestStackCount(t *testing.T) {
	var (
		size     = utils.RandInt(5) + 5
		expected = 3
	)

	fixedStk := NewFixedStack(size)
	for i := expected; i > 0; i-- {
		fixedStk.Push(expected)
	}

	assert.EqualValues(t, expected, fixedStk.Count(expected))
}

func TestStackContains(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"contains": func(t *testing.T) {
			var (
				size = 5
				val  = utils.RandInt(10)
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(val)
			assert.NoError(t, err)

			assert.True(t, fixedStk.Contains(val))
		},
		"not contains": func(t *testing.T) {
			var (
				size = 5
				val  = utils.RandInt(10)
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(val + utils.RandInt(5))
			assert.NoError(t, err)

			assert.False(t, fixedStk.Contains(val))
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestStackDump(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				size     = 5
				val      = utils.RandInt(10)
				expected = fmt.Sprintf("%v", []int{val})
			)

			fixedStk := NewFixedStack(size)
			err := fixedStk.Push(val)
			assert.NoError(t, err)

			assert.EqualValues(t, expected, fixedStk.Dump())
		},
		"empty stack": func(t *testing.T) {
			var (
				size     = 5
				expected = "empty stack"
			)

			fixedStk := NewFixedStack(size)
			assert.EqualValues(t, expected, fixedStk.Dump())
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}
