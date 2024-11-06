package recursion

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestEuclidean(t *testing.T) {
	timeout := 200 * time.Millisecond
	done := make(chan bool)

	go func() {
		x, y, expected := 22, 8, 2
		actual := Euclidean(x, y)
		assert.Equal(t, expected, actual)
		done <- true
	}()

	select {
	case <-done:
		assert.True(t, true, "test completed within the timeout")
	case <-time.After(timeout):
		t.Error("test exceeded the time limit of", timeout)
	}
}

func TestHanoi(t *testing.T) {
	// 테스트라기 보다는 내부에서 출력되는 문자열 확인용
	TowerOfHanoi(4, 1, 3)
	HanoiNormal(4, 1, 2, 3)
}

func TestQuickSort(t *testing.T) {
	var (
		arr      = []int{23, 55, 1, 2, 0, 32, 99, 3}
		expected = []int{0, 1, 2, 3, 23, 32, 55, 99}
	)

	sa := NewSortableArray(arr)
	sa.QuickSort(0, len(arr)-1)
	assert.Equal(t, arr, expected)
}

func TestQuickSelect(t *testing.T) {
	var (
		arr      = []int{23, 55, 1, 2, 0, 32, 99, 3}
		nth      = 4
		expected = 23
	)

	sa := NewSortableArray(arr)
	actual := sa.QuickSelectLowest(nth, 0, len(arr)-1)
	assert.Equal(t, expected, actual)
}
