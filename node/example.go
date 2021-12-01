package node

// 연결 리스트
type Node struct {
	data     string
	nextNode *Node
}

func NewNode(data string) *Node {
	return &Node{
		data: data,
	}
}

func (n *Node) AddNextNode(nextNode *Node) {
	n.nextNode = nextNode
}

type LinkedList struct {
	firstNode *Node
}

func NewLinkedList(firstNode *Node) *LinkedList {
	return &LinkedList{
		firstNode: firstNode,
	}
}

func (l *LinkedList) Read(idx int) string {
	currentNode := l.firstNode // 당연한거지만, 검색 시작은 언제나 첫 노드부터
	currentIdx := 0            // 이것도 당연히 첫 시작이니까 0부터

	for currentIdx < idx { // 찾고 있는 인덱스에 도달할 때까지 노드의 링크를 계속 따라들어감.
		currentNode = currentNode.nextNode
		currentIdx++

		if currentNode == nil { // 만약 현재 노드가 nil 이라면 리스트 끝에 도달했다는 거고,
			return "" // 찾고자 하는 인덱스의 노드를 못 찾았다는 뜻이므로 제로값 반환
		}
	}

	// currentIdx == idx 일 때 for문을 빠져 나왔을테니까
	// 찾은 것이므로 데이터 리턴
	return currentNode.data
}

func (l *LinkedList) IndexOf(data string) int {
	currentNode := l.firstNode
	currentIdx := 0

	for currentNode != nil {
		if currentNode.data == data { // 찾는 데이터를 가진 노드가 있으면
			return currentIdx // 인덱스 반환하면서 종료
		}

		// 못 찾았으면 다음 노드로 감.
		currentNode = currentNode.nextNode
		currentIdx++
	}

	// 마지막까지 찾지 못했을 경우
	return -1
}

func (l *LinkedList) InsertAt(idx int, data string) {
	currentNode := l.firstNode
	currentIdx := 0

	for currentIdx < idx { // 삽입하려는 인덱스의 바로 앞 노드를 찾음.
		currentNode = currentNode.nextNode
		currentIdx++

		if currentNode.nextNode == nil { // 만약 지정하려는 인덱스가 총 인덱스를 벗어났을 경우
			break
		}
	}

	// 바로 앞 노드를 찾았으면 노드 생성 후, 바로 앞 노드가 원래 갖고 있었던 다음 노드 주소 등록
	node := NewNode(data)
	node.AddNextNode(currentNode.nextNode)

	// 바로 앞 노드가 새로 생성된 노드를 참조하도록 변경
	currentNode.nextNode = node
}
