package queue

import "github.com/pkg/errors"

func TestQueue() error {
	testSet := []int{0, 78, 9, 45, 17, 0}
	capa := 11
	que := NewFixedQueue(capa)

	// test enqueue
	for _, v := range testSet {
		if err := que.EnQueue(v); err != nil {
			return errors.Errorf("enqueue err: %v", err)
		}
	}

	if capa == que.Len() {
		return errors.Errorf("not matched length")
	}

	// test dequeue
	for i := 0; i < 2; i++ {
		v, err := que.DeQueue()
		if err != nil {
			return errors.Errorf("dequeue err: %v", err)
		}
		println("dequeue result:", v)
	}

	que.Dump() // 앞에서 dequeue 를 2번 했기 때문에, 인덱스가 2부터 시작함.

	// test methods
	next, err := que.Peek()
	if err != nil {
		return errors.Errorf("peek err: %v", err)
	}
	println("peek result:", next)

	idx, err := que.Find(17)
	if err != nil {
		return errors.Errorf("find err: %v", err)
	}
	println("find result:", idx)

	cnt, err := que.Count(0)
	if err != nil {
		return errors.Errorf("count err: %v", err)
	}
	println("count result:", cnt)

	println("contains result:", que.Contains(7))

	// test dump after clear
	que.Clear()
	que.Dump() // 값이 없어서 아무것도 출력 안 됨.

	println("// start print manager\n")
	pm := NewPrintManager()
	pm.addJob("start Neunwelt Tactic!")
	pm.addJob("first, Kaede")
	pm.addJob("second, Miriam")
	pm.addJob("...")
	pm.addJob("last, riri")
	pm.addJob("shoot the magie sphere!\n")
	pm.run()

	return nil
}
