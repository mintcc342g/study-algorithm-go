package main

import (
	"main/array"
	"main/graph"

	"main/node"
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

	if err := node.TestNode(); err != nil {
		println(err.Error())
		return
	}

	if err := graph.TestGraph(); err != nil {
		println(err.Error())
		return
	}
}
