package node

import "errors"

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

	return nil
}
