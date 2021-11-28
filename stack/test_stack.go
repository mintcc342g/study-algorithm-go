package stack

func TestStack() error {
	// stack

	testSet := []int{3, 10, 2, 6, 1}
	stk := NewFixedStack(len(testSet))

	// test push
	for _, v := range testSet {
		if err := stk.Push(v); err != nil {
			return err
		}
	}

	println(stk.Dump())
	println(stk.Peek())

	stk.Pop()
	println(stk.Peek())

	stk.Clear()

	// test push again after empty
	testSet = []int{3, 10, 1, 6, 1}
	for _, v := range testSet {
		if err := stk.Push(v); err != nil {
			return err
		}
	}

	println(stk.Count(1))
	println(stk.Contains(10))
	println(stk.Contains(20))

	code := "l := []string{1,2,3}"
	linter := NewLinter()
	if err := linter.Lint(code); err != nil {
		return err
	}

	return nil
}
