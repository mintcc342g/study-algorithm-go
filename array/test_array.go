package array

func TestArray() error {

	println("\n// Start Test Array")

	sortedArray := []int{1, 2, 3, 4, 5}
	if err := binarySearch(sortedArray, 3); err != nil {
		return err
	}

	arr := []int{2, 34, 31, 55, 56, 90, 3, 22}
	bubbleSort(arr)

	arr = []int{2, 34, 31, 55, 56, 90, 3, 22}
	selectionSort(arr)

	arr = []int{2, 34, 31, 55, 56, 90, 3, 22}
	insertionSort(arr)

	return nil
}
