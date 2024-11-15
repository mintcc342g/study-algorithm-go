package array

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQuickSort(t *testing.T) {
	var (
		l        = []int{2, 7, 4, 3, 8}
		expected = []int{2, 3, 4, 7, 8}
	)

	assert.EqualValues(t, expected, QuickSort(l))
}
