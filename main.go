package main

import (
	"main/stack"
)

func main() {

	stkSample := []int{3, 10, 2, 6, 1}
	stk := stack.NewFixedStack(len(stkSample))

	for _, v := range stkSample {
		if err := stk.Push(v); err != nil {
			print(err)
			return
		}
	}

	println(stk.Dump())
	println(stk.Peek())

	stk.Pop()
	println(stk.Peek())

	stk.Clear()

	stkSample = []int{3, 10, 1, 6, 1}
	for _, v := range stkSample {
		if err := stk.Push(v); err != nil {
			print(err)
			return
		}
	}

	println(stk.Count(1))
	println(stk.Contains(10))
	println(stk.Contains(20))
}
