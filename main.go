package main

import (
	"study-algorithm-go/array"
	"study-algorithm-go/graph"
	"study-algorithm-go/heap"

	"study-algorithm-go/node"
	"study-algorithm-go/queue"
	"study-algorithm-go/recursion"
	"study-algorithm-go/stack"
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

	if err := heap.TestHeap(); err != nil {
		println(err.Error())
		return
	}
}
