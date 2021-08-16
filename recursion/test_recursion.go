package recursion

func TestRecursion() error {
	println("Euclidean", Euclidean(22, 8))

	TowerOfHanoi(4, 1, 3)

	HanoiNormal(4, 1, 2, 3)

	return nil
}
