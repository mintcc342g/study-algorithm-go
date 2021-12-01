package node

import "errors"

func TestNode() error {

	println("// Start Test Node")

	n0 := NewNode("000")
	ll := NewLinkedList(n0)

	n1 := NewNode("111")
	n0.addNextNode(n1)

	n2 := NewNode("222")
	n1.addNextNode(n2)

	n3 := NewNode("333")
	n2.addNextNode(n3)

	n4 := NewNode("444")
	n3.addNextNode(n4)

	r := ll.read(2)
	if r == "" {
		return errors.New("not found the node")
	}
	println("got the node's data", r)

	return nil
}
