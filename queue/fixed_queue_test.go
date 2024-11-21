package queue

import (
	"study-algorithm-go/utils"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQueueLen(t *testing.T) {
	var (
		capa     = utils.RandInt(5)
		item     = utils.RandInt(5)
		expected = 1
	)

	fixedQ := NewFixedQueue(capa)
	fixedQ.EnQueue(item)

	assert.Equal(t, expected, fixedQ.Len())
}

func TestQueueEnQueue(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				capa     = utils.RandInt(5)
				expected = utils.RandInt(10)
			)

			fixedQ := NewFixedQueue(capa)
			err := fixedQ.EnQueue(expected)
			assert.NoError(t, err)

			actualVal, err := fixedQ.DeQueue()
			assert.NoError(t, err)
			assert.EqualValues(t, expected, actualVal)
		},
		"fail to push": func(t *testing.T) {
			var (
				capa        = 0
				expectedErr = "capacity over"
			)

			fixedQ := NewFixedQueue(capa)
			err := fixedQ.EnQueue(utils.RandInt(10))
			assert.ErrorContains(t, err, expectedErr)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestQueueDeQueue(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				capa     = 7
				fixedQ   = NewFixedQueue(capa)
				expected = []int{}
			)

			for i := 0; i < capa; i++ {
				input := utils.RandInt(10)
				expected = append(expected, input)

				err := fixedQ.EnQueue(input)
				assert.NoError(t, err)
			}

			for i := 0; i < len(expected); i++ {
				actual, err := fixedQ.DeQueue()
				assert.NoError(t, err)
				assert.EqualValues(t, expected[i], actual)
			}
		},
		"fail to pop": func(t *testing.T) {
			var (
				capa        = 5
				expectedVal = 0
				expectedErr = "empty queue"
			)

			fixedQ := NewFixedQueue(capa)
			actualVal, err := fixedQ.DeQueue()
			assert.ErrorContains(t, err, expectedErr)
			assert.EqualValues(t, expectedVal, actualVal)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestQueuePeek(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				capa     = 5
				expected = 10
			)

			fixedQ := NewFixedQueue(capa)
			err := fixedQ.EnQueue(expected)
			assert.NoError(t, err)

			actual, err := fixedQ.Peek()
			assert.NoError(t, err)
			assert.EqualValues(t, expected, actual)
		},
		"empty stack": func(t *testing.T) {
			var (
				capa        = 5
				expectedVal = 0
				expectedErr = "empty queue"
			)

			fixedQ := NewFixedQueue(capa)
			actualVal, err := fixedQ.DeQueue()
			assert.ErrorContains(t, err, expectedErr)
			assert.EqualValues(t, expectedVal, actualVal)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestQueueFind(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				capa        = 5
				dummySize   = 2
				input       = utils.RandIntRange(11, 20)
				expectedIdx = 2
			)

			fixeQ := NewFixedQueue(capa)
			for i := 0; i < dummySize; i++ {
				err := fixeQ.EnQueue(utils.RandInt(10))
				assert.NoError(t, err)
			}
			err := fixeQ.EnQueue(input)
			assert.NoError(t, err)

			actual, err := fixeQ.Find(input)
			assert.NoError(t, err)
			assert.EqualValues(t, expectedIdx, actual)
		},
		"fail to find": func(t *testing.T) {
			var (
				capa        = 5
				input       = utils.RandInt(10)
				nonexsist   = utils.RandIntRange(11, 20)
				expectedIdx = 0
				expectedErr = "not found"
			)

			fixeQ := NewFixedQueue(capa)
			err := fixeQ.EnQueue(input)
			assert.NoError(t, err)

			actualVal, err := fixeQ.Find(nonexsist)
			assert.ErrorContains(t, err, expectedErr)
			assert.EqualValues(t, expectedIdx, actualVal)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestQueueCount(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"success": func(t *testing.T) {
			var (
				capa     = 5
				value    = utils.RandIntRange(uint(capa)+1, 20)
				expected = 2
				fixedQ   = NewFixedQueue(capa)
			)

			loop := capa - expected
			for i := 0; i < loop; i++ {
				err := fixedQ.EnQueue(i)
				assert.NoError(t, err)
			}

			for i := 0; i < expected; i++ {
				err := fixedQ.EnQueue(value)
				assert.NoError(t, err)
			}

			actual, err := fixedQ.Count(value)
			assert.NoError(t, err)
			assert.EqualValues(t, expected, actual)
		},
		"fail to find": func(t *testing.T) {
			var (
				capa        = 5
				input       = utils.RandInt(10)
				expectedIdx = 0
				expectedErr = "not found"
			)

			fixeQ := NewFixedQueue(capa)
			err := fixeQ.EnQueue(input)
			assert.NoError(t, err)

			actualVal, err := fixeQ.Find(input + utils.RandInt(10))
			assert.ErrorContains(t, err, expectedErr)
			assert.EqualValues(t, expectedIdx, actualVal)
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestQueueContains(t *testing.T) {
	testCases := map[string]func(t *testing.T){
		"contains": func(t *testing.T) {
			var (
				size = 5
				val  = utils.RandInt(10)
			)

			fixedQ := NewFixedQueue(size)
			err := fixedQ.EnQueue(val)
			assert.NoError(t, err)

			assert.True(t, fixedQ.Contains(val))
		},
		"not contains": func(t *testing.T) {
			var (
				size = 5
				val  = utils.RandInt(10)
			)

			fixedQ := NewFixedQueue(size)
			err := fixedQ.EnQueue(val + utils.RandInt(5))
			assert.NoError(t, err)

			assert.False(t, fixedQ.Contains(val))
		},
	}
	for scenario, test := range testCases {
		t.Run(scenario, test)
	}
}

func TestQueueClear(t *testing.T) {
	var (
		size     = utils.RandInt(5)
		val      = utils.RandInt(5)
		expected = 0
	)

	fixedQ := NewFixedQueue(size)
	fixedQ.EnQueue(val)
	fixedQ.Clear()

	assert.EqualValues(t, expected, fixedQ.Len())
}
