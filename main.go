package main

import (
	"main/array"
	"main/queue"
	"main/recursion"
	"main/stack"
)

func main() {

	if err := stack.TestStack(); err != nil {
		println(err.Error())
		return
	}

	if err := queue.TestQueue(); err != nil {
		println(err.Error())
		return
	}

	if err := recursion.TestRecursion(); err != nil {
		println(err.Error())
		return
	}

	if err := array.TestArray(); err != nil {
		println(err.Error())
		return
	}
}
