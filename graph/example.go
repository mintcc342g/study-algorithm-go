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

	// 1) 초기 그래프를 만들고, 연결된 두 노드를 가리키는 pairs를
	// 트리에 합쳐서 다같이 연결을 시키는 과정임. pairs로는 전체가
	// 다 연결된 그래프를 만들 수 없기 때문에 초기 그래프에 합치는 것
	graph := makeSet(len(s))
	for _, inner := range pairs {
		graph = union(inner[0], inner[1], graph)
	}

	// 2-1) 문자열 s의 각 문자별 인덱스의 루트 노드를 찾음.
	// 그리고 같은 루트 노드를 갖는 문자끼리 배열로 묶어서 맵에 저장
	store := make(map[int][]byte)
	for i := 0; i < len(s); i++ {
		root := find(i, graph)
		store[root] = append(store[root], s[i])
	}

	// 2-2) 위에서 저장한 같은 루트를 바라보는 문자들을 오름차순 정렬
	for root, subset := range store {
		sort.Slice(subset, func(i, j int) bool { return subset[i] < subset[j] })
		store[root] = subset
	}

	// 3) 그리고 오름차순으로 정렬된 각 집합의 문자들을 순서대로 넣어줌.
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

// 초기 graph를 만들며, 항상 [0,1,2,3,...,n]의 상태로 나와서
// 모든 노드가 자기 자신을 부모로 가리키는 상태로 초기화됨.
func makeSet(size int) (set []int) {
	set = make([]int, size)
	for i := 0; i < size; i++ {
		set[i] = i
	}

	return
}

// 입력받은 graph에서 i와 j의 루트 노드를 찾고,
// i와 j의 루트 노드의 높이를 비교해서 둘을 합쳐줌.
func union(i, j int, graph []int) []int {
	ir, jr := find(i, graph), find(j, graph)

	if ir == jr {
		return graph
	}

	if ir > jr { // union of rank 이며, 높이(인덱스)가 큰 트리 밑으로 높이가 작은 트리를 넣음.
		graph[ir] = jr // 예를 들어, 이 경우 i의 높이(인덱스)가 더 크므로, j 트리를 하위 트리로 넣어주는 것임.
	} else {
		graph[jr] = ir
	}

	return graph
}

// find 함수는 graph 내에서 x의 루트 노드의 인덱스를 반환함.
// path compression 기법이 적용되어 있어서, for문을 돌면서
// 루트 노드를 찾는 것과 동시에, x 노드부터의 루트 노드까지의
// 모든 노드의 부모 노드를 루트 노드로 바꿈.
func find(x int, graph []int) int {
	for graph[x] != x { // 루트 노드에 도달할 때까지 경로 압축을 반복 (루트 노드가 항상 0인 것은 아니기에 이렇게 조건을 줌.)
		x, graph[x] = graph[x], graph[graph[x]] // 노드 x의 부모를 부모의 부모 노드로 설정함으로써 경로를 단축함.
	}

	return x
}
