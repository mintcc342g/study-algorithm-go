package heap

import "fmt"

func TestHeap() (err error) {

	println("\n///// Heap")

	minHeap := NewMinHeap(10)
	minHeap.Add(1)
	minHeap.Add(5)
	minHeap.Add(10)
	minHeap.Add(4)

	println("\n// Check MinHeap")
	for i, v := range minHeap.elements {
		println(fmt.Sprintf("i: %d, v: %d", i, v))
	}

	minHeap.Pop()
	println("\n// Check MinHeap Pop")
	for i, v := range minHeap.elements {
		println(fmt.Sprintf("i: %d, v: %d", i, v))
	}

	minHeap.Add(6)
	println("\n// Check MinHeap's last value after adding a new value")
	for i, v := range minHeap.elements {
		println(fmt.Sprintf("i: %d, v: %d", i, v))
	}

	return nil
}
