package main

import (
	"main/queue"
	"main/stack"
)

func main() {

	// stack

	testSet := []int{3, 10, 2, 6, 1}
	stk := stack.NewFixedStack(len(testSet))

	// test push
	for _, v := range testSet {
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

	// test push again after empty
	testSet = []int{3, 10, 1, 6, 1}
	for _, v := range testSet {
		if err := stk.Push(v); err != nil {
			print(err)
			return
		}
	}

	println(stk.Count(1))
	println(stk.Contains(10))
	println(stk.Contains(20))

	// queue
	testSet = []int{0, 78, 9, 45, 17, 0}
	capa := 11
	que := queue.NewFixedQueue(capa)

	for _, v := range testSet {
		if err := que.EnQueue(v); err != nil {
			println(err)
			return
		}
	}

	if capa == que.Len() {
		println("not matched length")
		return
	}

	for i := 0; i < 2; i++ {
		v, err := que.DeQueue()
		if err != nil {
			println("dequeue err", err)
			return
		}
		println("dequeue result:", v)
	}

	que.Dump() // 앞에서 dequeue 를 2번 했기 때문에, 인덱스가 2부터 시작함.

	next, err := que.Peek()
	if err != nil {
		println(err)
		return
	}
	println("peek result:", next)

	idx, err := que.Find(17)
	if err != nil {
		println("find error", err)
		return
	}
	println("find result:", idx)

	cnt, err := que.Count(0)
	if err != nil {
		println("count error", err)
		return
	}
	println("count result:", cnt)

	println("contains result:", que.Contains(7))

	que.Clear()
	que.Dump() // 값이 없어서 아무것도 출력 안 됨.
}
