package recursion

import "fmt"

func TestRecursion() error {
	println("Euclidean", Euclidean(22, 8))

	TowerOfHanoi(4, 1, 3)

	HanoiNormal(4, 1, 2, 3)

	println("// start quickSort")
	arr := []int{23, 55, 1, 2, 0, 32, 99, 3}
	println("before quick sort", fmt.Sprintf("%v", arr))

	sa := NewSortableArray(arr)
	sa.quickSort(0, len(arr)-1)
	println("after quick sort", fmt.Sprintf("%v\n", sa.arr))

	println("// start quickSelect")
	arr = []int{23, 55, 1, 2, 0, 32, 99, 3}
	println("before quick select", fmt.Sprintf("%v", arr))

	sa = NewSortableArray(arr)
	println("quick select result", sa.quickSelectLowest(4, 0, len(arr)-1))
	println("after quick select", fmt.Sprintf("%v", arr))

	return nil
}
