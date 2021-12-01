package node

import (
	"errors"
	"fmt"
)

func TestNode() (err error) {

	println("// Start Test Node")

	n0 := NewNode("000")
	ll := NewLinkedList(n0)

	n1 := NewNode("111")
	n0.AddNextNode(n1)

	n2 := NewNode("222")
	n1.AddNextNode(n2)

	n3 := NewNode("333")
	n2.AddNextNode(n3)

	n4 := NewNode("444")
	n3.AddNextNode(n4)

	err = errors.New("not found the node")

	r := ll.Read(2)
	if r == "" {
		return
	}
	println("got the node's data", r)

	i := ll.IndexOf("333")
	if i < 0 {
		return
	}
	println("got the node's idx", i)

	ll.InsertAt(2, "151515")
	for i := 0; i < 5; i++ {
		r = ll.Read(i)
		println("check inserted values after insert", r)
	}

	i = 1
	println(fmt.Sprintf("node %d will be deleted", i))
	ll.DeleteAt(i)
	for i := 0; i < 5; i++ {
		r = ll.Read(i)
		println("check values after remove", r)
	}

	println("\n// start double linked list test using queue")

	q := NewQueue()
	q.EnQueue("one node")
	println("queue's tail is", q.Tail())

	q.EnQueue("two node")
	q.EnQueue("three node")
	println("after enqueue, queue's tail is", q.Tail())

	q.DeQueue()
	println("after dequeue, queue's head is", q.queue.ReadFirst())

	println("\n// start tree node test")

	tn := NewTreeNode(3)
	tn.Insert(2, tn)
	tn.Insert(4, tn)
	tn.Insert(1, tn)
	tn.Insert(5, tn)

	println("before delete tree node")
	tn.TraverseAndPrint(tn)

	tn.Delete(3, tn)

	println("after delete tree node")
	tn.TraverseAndPrint(tn)

	return nil
}
