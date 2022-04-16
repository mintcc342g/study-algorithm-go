package graph

import (
	"fmt"
	"sort"
)

/**
 * eg) BFS, DFS
 */

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

/*
 * leetCode : Smallest String With Swaps
 */

func smallestStringWithSwaps(s string, pairs [][]int) string {
	/* pairs 는 s 내의 각 글자들 중 연결된 애들을 알려줌.
	 * 연결된 애들 끼리는 오름차순이 완성될 때까지 바꿔도 됨.
	 * 즉,
	 * 1) 연결된 애들 끼리를 집합으로 만들고,
	 * 2) 각 집합 내에서 오름차순 정렬을 끝낸 후,
	 * 3) 그걸 다시 순서대로 합쳐주면 됨.
	 * 순서대로라는 건, s를 배열이라고 했을 때,
	 * 0번부터 n번까지의 인덱스 순서를 말하는 거임.
	 */

	// 1)
	graph := makeSet(len(s))
	for _, inner := range pairs {
		graph = union(inner[0], inner[1], graph)
	}

	// 2-1) 각 집합끼리 정렬 하려고 map 에 집합별로 저장해줌.
	store := make(map[int][]byte)
	for i := 0; i < len(s); i++ {
		root := find(i, graph)
		store[root] = append(store[root], s[i])
	}

	// 2-2) 집합 별로 오름차순 정렬시킴.
	for root, subset := range store {
		sort.Slice(subset, func(i, j int) bool { return subset[i] < subset[j] })
		store[root] = subset
	}

	// 3)
	result := []byte{}
	for i := 0; i < len(s); i++ {
		root := find(i, graph)
		subset, ok := store[root]
		if ok {
			result[i], store[root] = subset[0], subset[1:]
		}
	}

	return string(result)

}

func makeSet(size int) (set []int) {
	set = make([]int, size)
	for i := 0; i < size; i++ {
		set[i] = i
	}

	return
}

func union(i, j int, graph []int) []int {
	ir, jr := find(i, graph), find(j, graph)

	if ir == jr {
		return graph
	}

	if ir > jr {
		graph[ir] = jr
	} else {
		graph[jr] = ir
	}

	return graph
}

func find(x int, graph []int) int {
	for graph[x] != x {
		x, graph[x] = graph[x], graph[graph[x]]
	}

	return x
}
