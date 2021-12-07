package bfsdfs

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

func (p *Person) AddFriends(person ...*Person) {
	p.friends = append(p.friends, person...)
}

func (p *Person) DisplayNetwork() {

	reset := []*Person{p} // 검색 종료 후, 방문 기록을 전부 초기화 시키기 위한 변수

	queue := []*Person{p} // 방문 시작 때엔 루트 정점이 큐에 들어감. 이걸 배열 말고 이중 연결 리스트로 써도 될 듯.
	p.visited = true      // 방문 했다는 걸 표시

	var vertex *Person
	for len(queue) != 0 {
		vertex, queue = queue[0], queue[1:] // 현재 정점 설정하면서 큐에서 빼줌.
		println("visited", vertex.name)

		// 현재 정점의 인접 정점을 방문. 방문 안 했던 친구라면 모두 큐에 추가
		for _, friend := range vertex.friends {
			if !friend.visited {
				friend.visited = true
				queue = append(queue, friend)
				reset = append(reset, friend) // 검색 종료 후, 탐색 기록을 초기화 시키려고 넣어줌.
			}
		}
	}

	// 방문 기록 초기화
	for _, i := range reset {
		i.visited = false
	}
}

func (p *Person) DisplayNetworkByDfs() {

	reset := []*Person{p} // 검색 종료 후, 방문 기록을 전부 초기화 시키기 위한 변수

	stack := []*Person{p} // 방문 시작 때엔 루트 정점이 스택에 들어감.
	p.visited = true      // 방문 했다는 걸 표시

	var vertex *Person
	for len(stack) != 0 {
		vertex, stack = stack[len(stack)-1], stack[:len(stack)-1] // 현재 정점 설정하면서 스택에서 빼줌.
		println("visited", vertex.name)

		// 현재 정점의 인접 정점을 방문. 방문 안 했던 친구라면 모두 스택에 추가
		for _, friend := range vertex.friends {
			if !friend.visited {
				friend.visited = true
				stack = append(stack, friend)
				reset = append(reset, friend) // 검색 종료 후, 탐색 기록을 초기화 시키려고 넣어줌.
			}
		}
	}

	// 방문 기록 초기화
	for _, i := range reset {
		i.visited = false
	}
}
