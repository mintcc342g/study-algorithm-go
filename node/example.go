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

// 이 예제는 입력한 idx 위치의 +1 위치로 들어감.
// 2에 넣으려고 하면, 2랑 3사이에 넣는 것으로 보고, 실제로는 3번 인덱스에 들어가게 되는 것
func (l *LinkedList) InsertAt(idx int, data string) {
	currentNode := l.firstNode
	currentIdx := 0

	for currentIdx < idx { // 삽입하려는 인덱스에 있는 노드를 찾음.
		currentNode = currentNode.nextNode
		currentIdx++

		if currentNode.nextNode == nil { // 총 인덱스를 벗어났을 경우
			break
		}
	}

	// 노드를 찾았으면 새 노드 생성 후,
	node := NewNode(data)

	// 새 노드의 다음 노드값을 삽입하려는 위치에 있던 노드의 다음 노드 값으로 넣어줌.
	node.AddNextNode(currentNode.nextNode)

	// 삽입하려는 위치에 있던 노드가 새로 생성된 노드를 참조하도록 변경
	currentNode.nextNode = node
}

func (l *LinkedList) DeleteAt(idx int) {
	currentNode := l.firstNode
	currentIdx := 0

	for currentIdx < idx-1 { // 삭제하려는 인덱스의 바로 앞 노드를 찾음.
		currentNode = currentNode.nextNode // 현재 노드 값에는 삭제하려는 노드가 들어갈 것
		currentIdx++

		if currentNode.nextNode == nil { // 총 인덱스를 벗어났을 경우
			break
		}
	}

	nextNode := currentNode.nextNode.nextNode // 삭제하려는 노드의 바로 뒤 노드를
	currentNode.nextNode = nextNode           // 삭제하려는 노드의 앞 노드의 nextNode 값으로 넣어줌.
}

// 이중 연결 리스트 예제를 위한 노드
type DoubleLinkNode struct {
	prevNode *DoubleLinkNode
	nextNode *DoubleLinkNode
	data     string
}

func NewDoubleLinkNode(data string) *DoubleLinkNode {
	return &DoubleLinkNode{
		data: data,
	}
}

// 이중 연결 리스트
type DoubleLinkedList struct {
	firstNode *DoubleLinkNode
	lastNode  *DoubleLinkNode
}

func NewDoubleLinkedList(firstNode, lastNode *DoubleLinkNode) *DoubleLinkedList {
	return &DoubleLinkedList{
		firstNode: firstNode,
		lastNode:  lastNode,
	}
}

func (l *DoubleLinkedList) InsertAtEnd(data string) {
	node := NewDoubleLinkNode(data)

	if l.firstNode == nil { // 이중 연결 리스트가 비어 있을 때
		l.firstNode = node // 새로 만든 노드가 첫 노드이자 마지막 노드(밑에서 넣음)
	} else {
		node.prevNode = l.lastNode // 현재의 마지막 노드를 새로 만든 노드의 이전 노드로 만듦.
		l.lastNode.nextNode = node // 현재의 마지막 노드의 다음 노드를 새로 만든 노드로 넣음.
	}
	l.lastNode = node // 마지막 노드를 지금 만든 노드로 바꿔줌.
}

func (l *DoubleLinkedList) RemoveFromFront() (r string) {
	r = l.firstNode.nextNode.data      // 이건 보통 큐에서 데이터 빼서 사용하니까 해준 것
	l.firstNode = l.firstNode.nextNode // 삭제는 첫 번째 노드의 다음 노드를 넣어주기만 하면 됨.
	return
}

// 이하 Read* 메소드들은 값 제대로 들어갔는지 확인하려고 만든 것
func (l *DoubleLinkedList) ReadLast() string {
	return l.lastNode.data
}

func (l *DoubleLinkedList) ReadFirst() string {
	return l.firstNode.data
}

// 이중 연결 리스트로 구현한 큐 예제
type Queue struct {
	queue *DoubleLinkedList
}

func NewQueue() *Queue {
	return &Queue{
		queue: NewDoubleLinkedList(nil, nil),
	}
}

func (q *Queue) EnQueue(data string) {
	q.queue.InsertAtEnd(data)
}

func (q *Queue) DeQueue() string {
	return q.queue.RemoveFromFront()
}

func (q *Queue) Tail() string {
	return q.queue.ReadLast()
}

func (q *Queue) Head() string {
	return q.queue.ReadFirst()
}

// 이진 트리를 위한 노드
type TreeNode struct {
	left  *TreeNode // 왼쪽 자식
	right *TreeNode // 오른쪽 자식은 항상 부모보다 값이 커야한다는 게 이진 트리의 규칙
	data  int
}

func NewTreeNode(data int) *TreeNode {
	return &TreeNode{
		data: data,
	}
}

func (t *TreeNode) Search(data int, node *TreeNode) *TreeNode { // 노드를 받는 건 검색 시 루트 부터 시작 & 재귀를 위해서임.
	if node == nil || node.data == data {
		return node
	} else if data < node.data { // 부모 노드의 값보다 작으면 왼쪽 검색
		return t.Search(data, node.left)
	} else if data > node.data { // 부모 노드의 값보다 크면 오른쪽 검색
		return t.Search(data, node.right)
	}
	return nil
}

func (t *TreeNode) Insert(data int, node *TreeNode) {
	if data < node.data { // 왼쪽 검색 시작
		if node.left == nil { // 자식 없는 노드라면, 찾던 위치에 도달한 것이므로,
			node.left = NewTreeNode(data) // 자식 노드 생성해서 넣어줌.
		} else { // 그게 아니라면, 다시 검색 시작
			t.Insert(data, node.left)
		}
	} else if data > node.data { // 오른쪽 검색 시작. 원리는 위랑 같음.
		if node.right == nil {
			node.right = NewTreeNode(data)
		} else {
			t.Insert(data, node.right)
		}
	}
}

func (t *TreeNode) Delete(data int, node *TreeNode) *TreeNode {
	if node == nil { // 트리 최하단에 도착한 것이므로 재귀 종료
		return nil
	} else if data < node.data { // 삭제하려는 데이터가 왼쪽 트리..
		node.left = t.Delete(data, node.left)
		return node
	} else if data > node.data { // 삭제하려는 데이터가 오른쪽 트리..
		node.right = t.Delete(data, node.right)
		return node
	} else if data == node.data { // 삭제하려는 노드 찾았음.
		if node.left == nil { // 삭제 대상 노드의 왼쪽 자식 없으면,
			// 오른쪽 자식을 반환
			return node.right
			// 이렇게 해서 삭제 대상 노드가 삭제되고,
			// 오른쪽 노드가 후속자로 올라가서
			// 삭제 대상 노드의 부모의 왼쪽 자식으로 들어갈 것

		} else if node.right == nil { // 삭제 대상 노드의 오른쪽 자식이 없으면,
			// 그대로 왼쪽 자식을 반환하면 됨.
			return node.left
		}

		// 만약 양쪽에 자식이 있는 경우엔,
		// 현재 대상 노드를 삭제하고,
		// 그 자리를 후속자 노드로 대체함.
		node.right = t.lift(node.right, node)
		return node
	}

	return nil
}

func (t *TreeNode) lift(node, delNode *TreeNode) *TreeNode {
	// 현재 노드에 왼쪽 자식이 있으면,
	if node.left != nil {
		// 왼쪽 하위 트리를 계속 내려가서 후속자를 찾음.
		node.left = t.lift(node.left, delNode)
		return node
	}

	// 현재 노드의 오른쪽에 자식이 있건 말건,
	// 일단 왼쪽에 자식이 '없'다면,
	// 이 함수의 현재 노드가 후속자 노드라는 뜻이 되고,
	// 현재 노드의 값을 삭제 대상 노드의 새로운 값을 할당함???
	delNode.data = node.data // 후속자 노드의 오른쪽 자식이 부모의 왼쪽 자식으로 쓰일 수 있도록 반환함.
	return node.right
}

// 이진 트리의 중위 순회
// 트리에 있는 값을 전부 방문하면서 순서대로 출력해줌.
func (t *TreeNode) TraverseAndPrint(node *TreeNode) {
	if node == nil {
		return
	}
	node.TraverseAndPrint(node.left)
	println(node.data)
	node.TraverseAndPrint(node.right)
}
