package linkedlists

// 연결 리스트
type Node struct {
	data     string
	nextNode *Node
}

func newNode(data string) *Node {
	return &Node{
		data: data,
	}
}

func (n *Node) addNextNode(nextNode *Node) {
	n.nextNode = nextNode
}

type LinkedList struct {
	firstNode *Node
}

func NewLinkedList(data string) *LinkedList {
	return &LinkedList{
		firstNode: newNode(data),
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
	// InsertAt과 다르게 0부터 시작한 건, 이쪽의 리턴값이 currentNode이기 때문
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

// 예시라서 중복 데이터는 고려 안 함.
func (l *LinkedList) InsertAt(idx int, data string) {
	currentNode := l.firstNode
	currentIdx := 1 // 책에서는 0부터 시작하지만, firstNode는 무조건 존재하고, 값은 nextNode부터 들어가므로 1부터 시작해야 함.

	for currentIdx < idx { // 삽입하려는 인덱스에 있는 노드를 찾음.
		if currentNode == nil { // 인덱스를 벗어난 경우
			return
		}

		currentNode = currentNode.nextNode
		currentIdx++
	}

	node := newNode(data)                  // 노드를 찾았으면 새 노드 생성 후,
	node.addNextNode(currentNode.nextNode) // 새 노드의 다음 노드값을 삽입하려는 위치에 있던 노드의 다음 노드 값으로 넣어줌.
	currentNode.nextNode = node            // 삽입하려는 위치에 있던 노드가 새로 생성된 노드를 참조하도록 변경
}

func (l *LinkedList) DeleteAt(idx int) {
	currentNode := l.firstNode
	currentIdx := 0

	for currentIdx < idx-1 { // 삭제하려는 인덱스의 바로 앞 노드를 찾음.
		currentNode = currentNode.nextNode // 현재 노드 값에는 삭제하려는 노드가 들어갈 것
		currentIdx++

		if currentNode == nil { // 총 인덱스를 벗어났을 경우
			return
		}
	}

	nextNode := currentNode.nextNode.nextNode // 삭제하려는 노드의 바로 뒤 노드를
	currentNode.nextNode = nextNode           // 삭제하려는 노드의 앞 노드의 nextNode 값으로 넣어줌.
}

// leetcode
func (l *LinkedList) DeleteNthFromEnd(idx int) {

	stock := []*Node{}
	currentNode := l.firstNode

	for currentNode != nil {
		stock = append(stock, currentNode)
		currentNode = currentNode.nextNode
	}

	prev := len(stock) - idx - 1
	if prev == -1 {
		l.firstNode = l.firstNode.nextNode
	} else {
		currentNode = stock[prev]
		currentNode.nextNode = currentNode.nextNode.nextNode
	}
}

func (l *LinkedList) ReadAll() {
	currentNode := l.firstNode
	for currentNode != nil {
		println("ReadAll", "node.data", currentNode.data)
		currentNode = currentNode.nextNode
	}
}

/*
 ** Linked List의 병합 정렬
 */
func mergeSort(head *Node) *Node {
	if head == nil || head.nextNode == nil {
		return head
	}

	// 1. 중간 노드 찾기
	mid := findMiddle(head)

	// 2. 중간을 기준으로 리스트 나누기
	left := head
	right := mid.nextNode
	mid.nextNode = nil // left가 중간까지만 가도록 nil로 넣어주는 것

	// 3. 두 부분 리스트를 각각 정렬
	left = mergeSort(left)
	right = mergeSort(right)

	// 4. 정렬된 리스트를 병합
	return mergeTwoLists(left, right)
}

func findMiddle(head *Node) *Node {
	slow, fast := head, head
	for fast != nil && fast.nextNode != nil {
		slow = slow.nextNode
		fast = fast.nextNode.nextNode
	}
	return slow
}

func mergeTwoLists(l1, l2 *Node) *Node {
	dummy := &Node{}
	curr := dummy

	for l1 != nil && l2 != nil {
		if l1.data < l2.data {
			curr.nextNode = l1
			l1 = l1.nextNode
		} else {
			curr.nextNode = l2
			l2 = l2.nextNode
		}
		curr = curr.nextNode
	}

	if l1 != nil { // l1이 뒤에 더 남은 경우 붙여줌.
		curr.nextNode = l1
	} else { // l2가 뒤에 더 남은 경우 붙여줌.
		curr.nextNode = l2
	}

	return dummy.nextNode
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

func (l *DoubleLinkedList) ReadLastNext() string {
	return l.lastNode.nextNode.data
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
