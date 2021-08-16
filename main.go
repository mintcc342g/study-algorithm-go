package main

import (
	"main/queue"
	"main/recursion"
	"main/stack"
)

func main() {

	if err := stack.TestStack(); err != nil {
		println(err)
		return
	}

	if err := queue.TestQueue(); err != nil {
		println(err)
		return
	}

	if err := recursion.TestRecursion(); err != nil {
		println(err)
		return
	}
}
