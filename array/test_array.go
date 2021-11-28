package array

func TestArray() error {

	sortedArray := []int{1, 2, 3, 4, 5}
	if err := binarySearch(sortedArray, 3); err != nil {
		return err
	}

	return nil
}
