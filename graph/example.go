package graph

import "fmt"

type Person struct {
	name    string
	friends []*Person // 책에는 string 으로 되어 있음. 가중 그래프랑 차이를 보이려고 그렇게 한 듯?
	visited bool
}

func NewPerson(name string) *Person {
	return &Person{
		name: name,
	}
}

func (p *Person) AddFriends(friends ...*Person) {
	p.friends = append(p.friends, friends...)
}

func (p *Person) BFS() {
	queue := p.friends

	for len(queue) != 0 {
		node := queue[0]
		if !node.visited {
			node.visited = true
			queue = append(queue, node.friends...)
			println("BFS visited", fmt.Sprintf("%s", node.name))
		}
		queue = queue[1:]
	}
}

func (p *Person) DFS() {
	node := p
	stack := p.friends

	for len(stack) != 0 {
		node, stack = stack[len(stack)-1], stack[:len(stack)-1]
		println("DFS visited", fmt.Sprintf("%s", node.name))

		if len(node.friends) > 0 {
			stack = append(stack, node.friends...)
		}
	}
}
